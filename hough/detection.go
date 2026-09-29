package hough

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"sort"

	"gocv.io/x/gocv"
)

// for normalizing
const minDepth uint32 = 300 //mm
const maxDepth uint32 = 675 //mm

// circles with radii smaller than this will be ignored
const circleRThreshold = 18

type Circle struct {
	center image.Point
	radius int
}

// HoughConfig contains names for necessary resources (camera and vision service)
type HoughConfig struct {
	CameraName string `json:"camera_name"`

	Dp        float64 `json:"dp,omitempty"`
	MinDist   float64 `json:"min_dist,omitempty"`
	Param1    float64 `json:"param1,omitempty"`
	Param2    float64 `json:"param2,omitempty"`
	MinRadius int     `json:"min_radius,omitempty"`
	MaxRadius int     `json:"max_radius,omitempty"`
	Crop      *image.Rectangle
	SkipBlur  bool `json:"skip_blur"`
}

// Validate validates the config and returns implicit dependencies,
func (cfg *HoughConfig) Validate(path string) (requiredDeps, optionalDeps []string, err error) {
	if cfg.CameraName == "" {
		return nil, nil, fmt.Errorf(`expected "camera_name" attribute for object tracker %q`, path)
	}

	if cfg.Dp <= 0 {
		return nil, nil, fmt.Errorf("dp needs to be set (def 1)")
	}

	if cfg.MinDist <= 0 {
		return nil, nil, fmt.Errorf("min_dist needs to be set (def 8)")
	}

	if cfg.Param1 <= 0 {
		return nil, nil, fmt.Errorf("param1 needs to be set (def 60)")
	}

	if cfg.Param2 <= 0 {
		return nil, nil, fmt.Errorf("param2 needs to be set (def 25)")
	}

	if cfg.MinRadius <= 0 {
		return nil, nil, fmt.Errorf("min_radius needs to be set (def 35)")
	}

	if cfg.MaxRadius <= 0 {
		return nil, nil, fmt.Errorf("max_radius needs to be set (def 50)")
	}

	return []string{cfg.CameraName}, nil, nil
}

func (hc *HoughConfig) setDefaults() {
	hc.Dp = 1
	hc.MinDist = 8
	hc.Param1 = 60
	hc.Param2 = 25
	hc.MinRadius = 35
	hc.MaxRadius = 50
}

func vesselCircles(img image.Image, hc *HoughConfig, addOffset, outputBlur bool, outputResults string) ([]Circle, error) {
	croppedImg := cropImage(img, hc.Crop)
	mat := imageToMat(croppedImg)
	defer mat.Close()

	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(mat, &gray, gocv.ColorBGRToGray)

	if !hc.SkipBlur { // Blur to reduce noise
		gocv.MedianBlur(gray, &gray, 15)
		if outputBlur {
			if ok := gocv.IMWrite("blurred.jpg", gray); !ok {
				return nil, errors.New("failed to save the output image")
			}
		}
	}

	circles := gocv.NewMat()
	defer circles.Close()

	// READ MORE ABOUT THIS HERE:
	// https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
	gocv.HoughCirclesWithParams(
		gray,               // src
		&circles,           // circles
		gocv.HoughGradient, // method - only HoughGradient is supported
		hc.Dp,              // dp: inverse ratio of the accumulator resolution to the image resolution
		hc.MinDist,         // minDist: minimum distance between the centers of detected circles (Question: how is distance calculated here?)
		hc.Param1,          // param1: the higher threshold for the canny edge detector
		hc.Param2,          // param2: the accumulator threshold for circle detection
		hc.MinRadius,       // minRadius of bounding circle
		hc.MaxRadius,       // maxRadius of bouding circle
	)

	goodCircles := make([]Circle, 0)
	for i := 0; i < circles.Cols(); i++ {
		circle := circles.GetVecfAt(0, i)
		center := image.Pt(int(circle[0]), int(circle[1]))
		radius := int(circle[2])
		if radius < circleRThreshold {
			continue
		}
		if outputResults != "" {
			gocv.Circle(&mat, center, radius, color.RGBA{255, 0, 0, 0}, 2)
		}

		if addOffset && hc.Crop != nil {
			// need to add the offset back so circle is returned with respect to original image
			center = center.Add(hc.Crop.Min)
		}

		goodCircles = append(goodCircles, Circle{center, radius})
	}

	if outputResults != "" {
		if ok := gocv.IMWrite(outputResults, mat); !ok {
			return nil, errors.New("failed to save the output image")
		}
	}

	// order the circles by radius
	sort.Slice(goodCircles, func(i, j int) bool {
		return goodCircles[i].radius > goodCircles[j].radius
	})
	return goodCircles, nil
}

func cropImage(src image.Image, crop *image.Rectangle) image.Image {
	if crop == nil {
		return src
	}
	// Create a new RGBA image with the size of the crop rectangle
	croppedImg := image.NewRGBA(image.Rect(0, 0, crop.Dx(), crop.Dy()))

	// Adjust the draw point to correctly position the cropped area
	draw.Draw(croppedImg, croppedImg.Bounds(), src, crop.Min, draw.Src)
	return croppedImg
}

func imageToMat(img image.Image) gocv.Mat {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	mat := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8UC3)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			// Convert from 0-65535 to 0-255
			mat.SetUCharAt(y, x*3, uint8(b>>8))
			mat.SetUCharAt(y, x*3+1, uint8(g>>8))
			mat.SetUCharAt(y, x*3+2, uint8(r>>8))
		}
	}

	return mat
}
