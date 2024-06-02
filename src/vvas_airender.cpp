/*
 * Copyright 2021 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <vvas/vvas_kernel.h>
#include <gst/vvas/gstinferencemeta.h>
#include <chrono>

#include "vvas_airender.hpp"

#include <sys/stat.h>
#include <unistd.h>

int log_level = LOG_LEVEL_WARNING;

using namespace cv;
using namespace std;


#define MAX_CLASS_LEN 1024
#define MAX_LABEL_LEN 1024
#define MAX_ALLOWED_CLASS 20
#define MAX_ALLOWED_LABELS 20

struct color
{
  unsigned int blue;
  unsigned int green;
  unsigned int red;
};

struct vvass_xclassification
{
  color class_color;
  char class_name[MAX_CLASS_LEN];
};

struct overlayframe_info
{
  VVASFrame *inframe;
  Mat image;
  Mat I420image;
  Mat NV12image;
  Mat lumaImg;
  Mat chromaImg;
  int y_offset;
};


using Clock = std::chrono::steady_clock;

struct vvas_xoverlaypriv
{
  float font_size;
  unsigned int font;
  int line_thickness;
  int y_offset;
  color label_color;
  char label_filter[MAX_ALLOWED_LABELS][MAX_LABEL_LEN];
  unsigned char label_filter_cnt;
  unsigned short classes_count;
  vvass_xclassification class_list[MAX_ALLOWED_CLASS];
  struct overlayframe_info frameinfo;
  int drawfps;
  int fps_interv;
  double fps;
  int framecount;
  Clock::time_point startClk;
};



/* Check if the given classification is to be filtered */
int
vvas_classification_is_allowed (char *cls_name, vvas_xoverlaypriv * kpriv)
{
  unsigned int idx;

  if (cls_name == NULL)
    return -1;

  for (idx = 0;
      idx < sizeof (kpriv->class_list) / sizeof (kpriv->class_list[0]); idx++) {
    if (!strcmp (cls_name, kpriv->class_list[idx].class_name)) {
      return idx;
    }
  }
  return -1;
}

/* Get y and uv color components corresponding to givne RGB color */
void
convert_rgb_to_yuv_clrs (color clr, unsigned char *y, unsigned short *uv)
{
  Mat YUVmat;
  Mat BGRmat (2, 2, CV_8UC3, Scalar (clr.red, clr.green, clr.blue));
  cvtColor (BGRmat, YUVmat, cv::COLOR_BGR2YUV_I420);
  *y = YUVmat.at < uchar > (0, 0);
  *uv = YUVmat.at < uchar > (2, 0) << 8 | YUVmat.at < uchar > (2, 1);
  return;
}

// Function to check if a file exists
bool fileExists(const std::string& filepath) {
    std::ifstream file(filepath);
    return file.good();
}

void saveMatToTextFile(const Mat& mat, const Rect& roi, const std::string& filename) {
    if (!fileExists(filename)) {
        std::ofstream file(filename);
        if (file.is_open()) {
            Mat roiMat = mat(roi); // Extract ROI from the mat
            for (int i = 0; i < roiMat.rows; ++i) {
                for (int j = 0; j < roiMat.cols; ++j) {
                    file << static_cast<int>(roiMat.at<uchar>(i, j)) << " ";
                }
                file << std::endl;
            }
            file.close();
            std::cout << "File " << filename << " created successfully." << std::endl;
        } else {
            std::cerr << "Unable to open file " << filename << std::endl;
        }
    }
	//else {
    //    std::cout << "File " << filename << " already exists." << std::endl;
    //}
}


void saveBGRMatToTextFile(const Mat& mat, const std::string& filename) {
    if (!fileExists(filename)) {
        std::ofstream file(filename);
        if (file.is_open()) {
            for (int i = 0; i < mat.rows; ++i) {
                for (int j = 0; j < mat.cols; ++j) {
                    file << static_cast<int>(mat.at<uchar>(i, j)) << " ";
                }
                file << std::endl;
            }
            file.close();
            std::cout << "File " << filename << " created successfully." << std::endl;
        } else {
            std::cerr << "Unable to open file " << filename << std::endl;
        }
    } 
	//else {
    //    std::cout << "File " << filename << " already exists." << std::endl;
    //}
}

void saveChromaMatToTextFile(const Mat& mat, const Rect& roi, const std::string& filename) {
    if (!fileExists(filename)) {
        std::ofstream file(filename);
        if (file.is_open()) {
            Mat roiMat = mat(roi); // Extract ROI from the mat
            for (int i = 0; i < roiMat.rows; ++i) {
                for (int j = 0; j < roiMat.cols; ++j) {
                    file << roiMat.at<uint16_t>(i, j) << " ";
                }
                file << std::endl;
            }
            file.close();
            std::cout << "File " << filename << " created successfully." << std::endl;
        } else {
            std::cerr << "Unable to open file " << filename << std::endl;
        }
    } 
	//else {
    //    std::cout << "File " << filename << " already exists." << std::endl;
    //}
}

/* Compose label text based on config json */
bool
get_label_text (GstInferenceClassification * c, vvas_xoverlaypriv * kpriv,
    char *label_string)
{
  unsigned char idx = 0, buffIdx = 0;
  if (!c->class_label || !strlen ((char *) c->class_label))
    return false;

  for (idx = 0; idx < kpriv->label_filter_cnt; idx++) {
    if (!strcmp (kpriv->label_filter[idx], "class")) {
      sprintf (label_string + buffIdx, "%s", (char *) c->class_label);
      buffIdx += strlen (label_string);
    } else if (!strcmp (kpriv->label_filter[idx], "probability")) {
      sprintf (label_string + buffIdx, " : %.2f ", c->class_prob);
      buffIdx += strlen (label_string);
    }
  }
  return true;
}

static gboolean
overlay_node_foreach (GNode * node, gpointer kpriv_ptr)
{
  vvas_xoverlaypriv *kpriv = (vvas_xoverlaypriv *) kpriv_ptr;
  struct overlayframe_info *frameinfo = &(kpriv->frameinfo);
  LOG_MESSAGE (LOG_LEVEL_DEBUG, "enter");

  GList *classes;
  GstInferenceClassification *classification;
  GstInferencePrediction *prediction = (GstInferencePrediction *) node->data;

  /* On each children, iterate through the different associated classes */
  for (classes = prediction->classifications;
      classes; classes = g_list_next (classes)) {
    classification = (GstInferenceClassification *) classes->data;

    int idx = vvas_classification_is_allowed ((char *)
        classification->class_label, kpriv);
    if (kpriv->classes_count && idx == -1)
      continue;

    color clr;
    if (kpriv->classes_count) {
      clr = {
      kpriv->class_list[idx].class_color.blue,
            kpriv->class_list[idx].class_color.green,
            kpriv->class_list[idx].class_color.red};
    } else {
      /* If there are no classes specified, we will go with default blue */
      clr = {
      255, 0, 0};
    }

    char label_string[MAX_LABEL_LEN];
	char new_label_string[MAX_LABEL_LEN];
    bool label_present;
    Size textsize;
    label_present = get_label_text (classification, kpriv, label_string);

    if (label_present) {
      int baseline;
      textsize = getTextSize (label_string, kpriv->font,
          kpriv->font_size, 1, &baseline);
      /* Get y offset to use in case of classification model */
      if ((prediction->bbox.height < 1) && (prediction->bbox.width < 1)) {
        if (kpriv->y_offset) {
          frameinfo->y_offset = kpriv->y_offset;
        } else {
          frameinfo->y_offset = (frameinfo->inframe->props.height * 0.10);
        }
      }
    }

    LOG_MESSAGE (LOG_LEVEL_INFO,
        "RESULT: (prediction node %ld) %s(%d) %d %d %d %d (%f)",
        prediction->prediction_id,
        label_present ? classification->class_label : NULL,
        classification->class_id, prediction->bbox.x, prediction->bbox.y,
        prediction->bbox.width + prediction->bbox.x,
        prediction->bbox.height + prediction->bbox.y,
        classification->class_prob);

    /* Check whether the frame is NV12 or BGR and act accordingly */
    if (frameinfo->inframe->props.fmt == VVAS_VFMT_Y_UV8_420) {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "Drawing rectangle for NV12 image");
      unsigned char yScalar;
      unsigned short uvScalar;
      convert_rgb_to_yuv_clrs (clr, &yScalar, &uvScalar);
      /* Draw rectangle on y an uv plane */
      int new_xmin = floor (prediction->bbox.x / 2) * 2;
      int new_ymin = floor (prediction->bbox.y / 2) * 2;
      int new_xmax =
          floor ((prediction->bbox.width + prediction->bbox.x) / 2) * 2;
      int new_ymax =
          floor ((prediction->bbox.height + prediction->bbox.y) / 2) * 2;
	  //CHRISTOS
	  int stop_dist;
	  
      Size test_rect (new_xmax - new_xmin, new_ymax - new_ymin);
	   
	  /* ------HERE------*/
      if (!(!prediction->bbox.x && !prediction->bbox.y)) {
        rectangle (frameinfo->lumaImg, Point (new_xmin,
              new_ymin), Point (new_xmax,
              new_ymax), Scalar (yScalar), kpriv->line_thickness, 1, 0);
        rectangle (frameinfo->chromaImg, Point (new_xmin / 2,
              new_ymin / 2), Point (new_xmax / 2,
              new_ymax / 2), Scalar (uvScalar), kpriv->line_thickness, 1, 0);
      }
	  
	  
      if (label_present) {
		/* ------HERE------ */
        /* Draw filled rectangle for labelling, both on y and uv plane */

		Mat u_plane, v_plane;

		int chroma_height = frameinfo->chromaImg.rows;
		int chroma_width = frameinfo->chromaImg.cols;
		int luma_height = frameinfo->lumaImg.rows;
		int luma_width = frameinfo->lumaImg.cols;

		// Initialize U and V planes with half the resolution of the luma plane
		u_plane.create(chroma_height, chroma_width, CV_8UC1);
		v_plane.create(chroma_height, chroma_width, CV_8UC1);

		for (int i = 0; i < chroma_height; ++i) {
			for (int j = 0; j < chroma_width; ++j) {
				uint16_t uv_value = frameinfo->chromaImg.at<uint16_t>(i, j);
				u_plane.at<uchar>(i, j) = uv_value & 0xFF; // Extract lower 8 bits for U
				v_plane.at<uchar>(i, j) = (uv_value >> 8) & 0xFF; // Extract upper 8 bits for V
			}
		}
		/*
		Mat yuv_img(luma_height + luma_height / 2, luma_width, CV_8UC1);
		memcpy(yuv_img.data, frameinfo->lumaImg.data, luma_width * luma_height);
    
		uchar* uv_ptr = yuv_img.data + luma_width * luma_height;
		for (int i = 0; i < luma_height / 2; ++i) {
			for (int j = 0; j < luma_width / 2; ++j) {
				uv_ptr[i * luma_width + 2 * j] = u_plane.at<uchar>(i, j);
				uv_ptr[i * luma_width + 2 * j + 1] = v_plane.at<uchar>(i, j);
			}
		}
		
		Mat bgr_img;
		cvtColor(yuv_img, bgr_img, COLOR_YUV2BGR_NV12);
		*/
		
		
/*			
		int width_luma = frameinfo->lumaImg.cols;
		int height_luma = frameinfo->lumaImg.rows;
	
		// Create a YUV image
		Mat yuv_img(height_luma + height_luma / 2, width_luma, CV_8UC1);
		memcpy(yuv_img.data, frameinfo->lumaImg.data, width_luma * height_luma);
		memcpy(yuv_img.data + width_luma * height_luma, frameinfo->chromaImg.data, width_luma * height_luma / 2);

		// Convert YUV to BGR
		Mat bgr_img;
		cvtColor(yuv_img, bgr_img, cv::COLOR_YUV2BGR_I420);
		
		std::vector<cv::Mat> bgr_channels;
		cv::split(bgr_img, bgr_channels);

		Scalar blue_sum = cv::mean(bgr_channels[0]);
		Scalar green_sum = cv::mean(bgr_channels[1]);
		Scalar red_sum = cv::mean(bgr_channels[2]);
*/
/*	
    // Create a single NV12 image
    cv::Mat nv12Img(height + height / 2, stride, CV_8UC1);

    // Copy luma to NV12 image
    frameinfo->lumaImg.copyTo(nv12Img(cv::Rect(0, 0, stride, height)));

    // Copy chroma to NV12 image (need to reshape chroma data)
    cv::Mat chromaReshaped(height / 2, stride, CV_8UC1, chromaBuf);
    chromaReshaped.copyTo(nv12Img(cv::Rect(0, height, stride, height / 2)));

    // Convert NV12 to BGR
    cv::Mat bgrImg;
    cv::cvtColor(nv12Img, bgrImg, cv::COLOR_YUV2BGR_NV12);
	*/

		if (idx == 1) {
			//This is idx for Stop Sign
			//std::sprintf(new_label_string, "Dy = %d and Dx = %d", new_ymax-new_ymin, new_xmax-new_xmin);
			std::sprintf(new_label_string, "C%dx%d", frameinfo->I420image.rows, frameinfo->I420image.cols);
			stop_dist = ceil(5*200/(new_ymax-new_ymin));
			std::sprintf(new_label_string, "IN < %d m", stop_dist);
			rectangle (frameinfo->lumaImg, Rect (Point (1000,
						1000 - textsize.height), textsize),
				Scalar (yScalar), FILLED, 1, 0);
			textsize.height /= 2;
			textsize.width /= 2;
			rectangle (frameinfo->chromaImg, Rect (Point (1000 / 2,
						1000 / 2 - textsize.height), textsize),
				Scalar (uvScalar), FILLED, 1, 0);
				
			/* Draw label text on the filled rectanngle */
			convert_rgb_to_yuv_clrs (kpriv->label_color, &yScalar, &uvScalar);
			putText (frameinfo->lumaImg, new_label_string, cv::Point (1000,
					1000 + frameinfo->y_offset), kpriv->font, kpriv->font_size,
				Scalar (yScalar), 1, 1);
			putText (frameinfo->chromaImg, new_label_string, cv::Point (1000 / 2,
					1000 / 2 + frameinfo->y_offset / 2), kpriv->font,
				kpriv->font_size / 2, Scalar (uvScalar), 1, 1);
		} else if (idx == 0) {
			/*
			// Define the region of interest (ROI)
			Rect roi(new_xmin, new_ymin, new_xmax, new_ymin + (new_ymax-new_ymin)/3); // Example ROI
			Mat roi_img = bgr_img(roi);

			// Extract the red channel and calculate the sum
			vector<cv::Mat> channels;
			split(roi_img, channels);
			Mat red_channel = channels[2];

			double red_mean = cv::mean(red_channel)[0];
			double red_sum = cv::sum(red_channel)[0];
			*/
			
			Rect roi(new_xmin, new_ymin, new_xmax, new_ymin + (new_ymax-new_ymin)/3); // Example ROI	
			Rect roi_2(new_xmin/2, new_ymin/2, new_xmax/2, (new_ymin + (new_ymax-new_ymin)/3)/2); // Example ROI			
			
			//Rect roi(new_xmin, new_ymin + 2*(new_ymax-new_ymin)/3, new_xmax, new_ymax); // Example ROI	
			//Rect roi_2(new_xmin/2, (new_ymin + (new_ymax-new_ymin)/3)/2, new_xmax/2, new_ymax/2); // Example ROI			
			
			Mat rgbImg(roi.height, roi.width, CV_8UC3);
			int pixel_count = 0;
			int red_count = 0;
			long sum_red = 0;
			long sum_green = 0;
			long sum_blue = 0;
			for (int i = roi.y; i < roi.y + roi.height; ++i) {
				for (int j = roi.x; j < roi.x + roi.width; ++j) {
					uchar y = frameinfo->lumaImg.at<uchar>(i, j);
					uchar u = u_plane.at<uchar>(i / 2, j / 2);
					uchar v = v_plane.at<uchar>(i / 2, j / 2);

					uchar r, g, b;
					int c = y - 16;
					int d = u - 128;
					int e = v - 128;

					r = saturate_cast<uchar>(( 298 * c + 409 * e + 128) >> 8);
					g = saturate_cast<uchar>(( 298 * c - 100 * d - 208 * e + 128) >> 8);
					b = saturate_cast<uchar>(( 298 * c + 516 * d + 128) >> 8);

					Vec3b &pixel = rgbImg.at<Vec3b>(i - roi.y, j - roi.x);
					pixel[2] = r;
					pixel[1] = g;
					pixel[0] = b;
					if (r>200){
						++red_count;
					}
					++pixel_count;
				}
			}
			
			// Save lumaImg as text file
			saveMatToTextFile(frameinfo->lumaImg, roi, "lumaImg.txt");

			// Save chromaImg as text file
			saveChromaMatToTextFile(frameinfo->chromaImg, roi_2, "chromaImg.txt");
			
			// Save u_plane as text file
			saveMatToTextFile(u_plane, roi_2, "u_plane.txt");
			
			// Save v_plane as text file
			saveMatToTextFile(v_plane, roi_2, "v_plane.txt");
			
			// Save v_plane as text file
			saveBGRMatToTextFile(rgbImg, "rgbImg.txt");
			
			double mean_red = static_cast<double>(sum_red) / pixel_count;
			double mean_green = static_cast<double>(sum_green) / pixel_count;
			double mean_blue = static_cast<double>(sum_blue) / pixel_count;
			//std::sprintf(new_label_string, "R: %.1f,G: %.1f, B: %.1f", mean_red, mean_green, mean_blue);
			std::sprintf(new_label_string, "R: %d, Total: %d", red_count, pixel_count);
			rectangle (frameinfo->lumaImg, Rect (Point (new_xmin,
						new_ymin - textsize.height), textsize),
				Scalar (yScalar), FILLED, 1, 0);
			textsize.height /= 2;
			textsize.width /= 2;
			rectangle (frameinfo->chromaImg, Rect (Point (new_xmin / 2,
						new_ymin / 2 - textsize.height), textsize),
				Scalar (uvScalar), FILLED, 1, 0);

			/* Draw label text on the filled rectanngle */
			convert_rgb_to_yuv_clrs (kpriv->label_color, &yScalar, &uvScalar);
			putText (frameinfo->lumaImg, new_label_string, cv::Point (new_xmin,
					new_ymin + frameinfo->y_offset), kpriv->font, kpriv->font_size,
				Scalar (yScalar), 1, 1);
			putText (frameinfo->chromaImg, new_label_string, cv::Point (new_xmin / 2,
					new_ymin / 2 + frameinfo->y_offset / 2), kpriv->font,
				kpriv->font_size / 2, Scalar (uvScalar), 1, 1);
		} else {
			std::copy(new_label_string, new_label_string + 5, label_string);
		}
      }
    } else if (frameinfo->inframe->props.fmt == VVAS_VFMT_BGR8) {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "Drawing rectangle for BGR image");

      if (!(!prediction->bbox.x && !prediction->bbox.y)) {
        /* Draw rectangle over the dectected object */
        rectangle (frameinfo->image, Point (prediction->bbox.x,
              prediction->bbox.y),
          Point (prediction->bbox.width + prediction->bbox.x,
              prediction->bbox.height + prediction->bbox.y), Scalar (clr.blue,
              clr.green, clr.red), kpriv->line_thickness, 1, 0);
      }

      if (label_present) {
        /* Draw filled rectangle for label */ 
		/* NOT HERE */ 
		
        rectangle (frameinfo->image, Rect (Point (prediction->bbox.x,
                    prediction->bbox.y - textsize.height), textsize),
            Scalar (clr.blue, clr.green, clr.red), FILLED, 1, 0);

        /* Draw label text on the filled rectanngle */
        putText (frameinfo->image, label_string,
            cv::Point (prediction->bbox.x,
                prediction->bbox.y + frameinfo->y_offset), kpriv->font,
            kpriv->font_size, Scalar (kpriv->label_color.blue,
                kpriv->label_color.green, kpriv->label_color.red), 1, 1);
      }
    }
  }



  return FALSE;
}

static void
fps_overlay(gpointer kpriv_ptr)
{
  vvas_xoverlaypriv *kpriv = (vvas_xoverlaypriv *) kpriv_ptr;
  if (!kpriv->drawfps)
  {
      return ;
  }
  struct overlayframe_info *frameinfo = &(kpriv->frameinfo);

  if (kpriv->framecount == 0)
  {
      kpriv->startClk = Clock::now();
  }
  else 
  {
      if (kpriv->framecount%kpriv->fps_interv == 0)
      {

          Clock::time_point nowClk = Clock::now();
          int duration = (std::chrono::duration_cast<std::chrono::milliseconds>(nowClk - kpriv->startClk)).count();
          kpriv->fps = kpriv->framecount * 1e3 / duration ;
      }

      color clr = {255, 0, 0};
      int new_xmin = 50;
      int new_ymin = 50;

      std::ostringstream oss;
      oss  << "Framerate:" << kpriv->fps << " FPS";
      Size textsize;

      if (frameinfo->inframe->props.fmt == VVAS_VFMT_Y_UV8_420) {
          unsigned char yScalar;
          unsigned short uvScalar;
          convert_rgb_to_yuv_clrs (clr, &yScalar, &uvScalar);
          {
              /* Draw label text on the filled rectanngle */
              convert_rgb_to_yuv_clrs (kpriv->label_color, &yScalar, &uvScalar);
              putText (frameinfo->lumaImg, oss.str(), cv::Point (new_xmin,
                          new_ymin), kpriv->font, kpriv->font_size,
                      Scalar (yScalar), 1, 1);
              putText (frameinfo->chromaImg, oss.str(), cv::Point (new_xmin / 2,
                          new_ymin / 2), kpriv->font,
                      kpriv->font_size / 2, Scalar (uvScalar), 1, 1);
          }
      } else if (frameinfo->inframe->props.fmt == VVAS_VFMT_BGR8) {
          LOG_MESSAGE (LOG_LEVEL_DEBUG, "Drawing rectangle for BGR image");
          {
              /* Draw label text on the filled rectanngle */
              putText (frameinfo->image, oss.str(),
                      cv::Point (new_xmin, new_ymin), kpriv->font,
                      kpriv->font_size, Scalar (clr.blue,
                          clr.green, clr.red), 1, 1);
          }
      }
  }
  kpriv->framecount++;

  return ;
}

extern "C"
{
  int32_t xlnx_kernel_init (VVASKernel * handle)
  {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "enter");

    vvas_xoverlaypriv *kpriv =
        (vvas_xoverlaypriv *) calloc (1, sizeof (vvas_xoverlaypriv));

    json_t *jconfig = handle->kernel_config;
    json_t *val, *karray = NULL, *classes = NULL;

    /* Initialize config params with default values */
    log_level = LOG_LEVEL_WARNING;
    kpriv->font_size = 0.5;
    kpriv->font = 0;
    kpriv->line_thickness = 1;
    kpriv->y_offset = 0;
    kpriv->label_color = {0, 0, 0};
    strcpy(kpriv->label_filter[0], "class");
    strcpy(kpriv->label_filter[1], "probability");
    kpriv->label_filter_cnt = 2;
    kpriv->classes_count = 0;
    kpriv->framecount = 0;

    char* env = getenv("SMARTCAM_SCREENFPS");
    if (env)
    {
        kpriv->drawfps = 1;
    }
    else
    {
        kpriv->drawfps = 0;
    }

    val = json_object_get (jconfig, "fps_interval");
    if (!val || !json_is_integer (val))
        kpriv->fps_interv = 1;
    else
        kpriv->fps_interv = json_integer_value (val);

    val = json_object_get (jconfig, "debug_level");
    if (!val || !json_is_integer (val))
        log_level = LOG_LEVEL_WARNING;
    else
        log_level = json_integer_value (val);

      val = json_object_get (jconfig, "font_size");
    if (!val || !json_is_integer (val))
        kpriv->font_size = 0.5;
    else
        kpriv->font_size = json_integer_value (val);

      val = json_object_get (jconfig, "font");
    if (!val || !json_is_integer (val))
        kpriv->font = 0;
    else
        kpriv->font = json_integer_value (val);

      val = json_object_get (jconfig, "thickness");
    if (!val || !json_is_integer (val))
        kpriv->line_thickness = 1;
    else
        kpriv->line_thickness = json_integer_value (val);

      val = json_object_get (jconfig, "y_offset");
    if (!val || !json_is_integer (val))
        kpriv->y_offset = 0;
    else
        kpriv->y_offset = json_integer_value (val);

    /* get label color array */
      karray = json_object_get (jconfig, "label_color");
    if (!karray)
    {
      LOG_MESSAGE (LOG_LEVEL_ERROR, "failed to find label_color");
      return -1;
    } else
    {
      kpriv->label_color.blue =
          json_integer_value (json_object_get (karray, "blue"));
      kpriv->label_color.green =
          json_integer_value (json_object_get (karray, "green"));
      kpriv->label_color.red =
          json_integer_value (json_object_get (karray, "red"));
    }

    karray = json_object_get (jconfig, "label_filter");

    if (!json_is_array (karray)) {
      LOG_MESSAGE (LOG_LEVEL_ERROR, "label_filter not found in the config\n");
      return -1;
    }
    kpriv->label_filter_cnt = 0;
    for (unsigned int index = 0; index < json_array_size (karray); index++) {
      strcpy (kpriv->label_filter[index],
          json_string_value (json_array_get (karray, index)));
      kpriv->label_filter_cnt++;
    }

    /* get classes array */
    karray = json_object_get (jconfig, "classes");
    if (!karray) {
      LOG_MESSAGE (LOG_LEVEL_ERROR, "failed to find key labels");
      return -1;
    }

    if (!json_is_array (karray)) {
      LOG_MESSAGE (LOG_LEVEL_ERROR, "labels key is not of array type");
      return -1;
    }
    kpriv->classes_count = json_array_size (karray);
    for (unsigned int index = 0; index < kpriv->classes_count; index++) {
      classes = json_array_get (karray, index);
      if (!classes) {
        LOG_MESSAGE (LOG_LEVEL_ERROR, "failed to get class object");
        return -1;
      }

      val = json_object_get (classes, "name");
      if (!json_is_string (val)) {
        LOG_MESSAGE (LOG_LEVEL_ERROR, "name is not found for array %d", index);
        return -1;
      } else {
        strncpy (kpriv->class_list[index].class_name,
            (char *) json_string_value (val), MAX_CLASS_LEN - 1);
        LOG_MESSAGE (LOG_LEVEL_DEBUG, "name %s",
            kpriv->class_list[index].class_name);
      }

      val = json_object_get (classes, "green");
      if (!val || !json_is_integer (val))
        kpriv->class_list[index].class_color.green = 0;
      else
        kpriv->class_list[index].class_color.green = json_integer_value (val);

      val = json_object_get (classes, "blue");
      if (!val || !json_is_integer (val))
        kpriv->class_list[index].class_color.blue = 0;
      else
        kpriv->class_list[index].class_color.blue = json_integer_value (val);

      val = json_object_get (classes, "red");
      if (!val || !json_is_integer (val))
        kpriv->class_list[index].class_color.red = 0;
      else
        kpriv->class_list[index].class_color.red = json_integer_value (val);
    }

    handle->kernel_priv = (void *) kpriv;
    return 0;
  }

  uint32_t xlnx_kernel_deinit (VVASKernel * handle)
  {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "enter");
    vvas_xoverlaypriv *kpriv = (vvas_xoverlaypriv *) handle->kernel_priv;

    if (kpriv)
      free (kpriv);

    return 0;
  }


  uint32_t xlnx_kernel_start (VVASKernel * handle, int start,
      VVASFrame * input[MAX_NUM_OBJECT], VVASFrame * output[MAX_NUM_OBJECT])
  {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "enter");
    GstInferenceMeta *infer_meta = NULL;
    char *pstr;

    vvas_xoverlaypriv *kpriv = (vvas_xoverlaypriv *) handle->kernel_priv;
    struct overlayframe_info *frameinfo = &(kpriv->frameinfo);
    frameinfo->y_offset = 0;
    frameinfo->inframe = input[0];
    char *indata = (char *) frameinfo->inframe->vaddr[0];
    char *lumaBuf = (char *) frameinfo->inframe->vaddr[0];
    char *chromaBuf = (char *) frameinfo->inframe->vaddr[1];
    infer_meta = ((GstInferenceMeta *) gst_buffer_get_meta ((GstBuffer *)
            frameinfo->inframe->app_priv, gst_inference_meta_api_get_type ()));
    if (infer_meta == NULL) {
      LOG_MESSAGE (LOG_LEVEL_WARNING,
          "vvas meta data is not available for postdpu");
    } else {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "vvas_mata ptr %p", infer_meta);
    }

    if (frameinfo->inframe->props.fmt == VVAS_VFMT_Y_UV8_420) {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "Input frame is in NV12 format\n");
      frameinfo->lumaImg.create (input[0]->props.height, input[0]->props.stride,
          CV_8UC1);
      frameinfo->lumaImg.data = (unsigned char *) lumaBuf;
      frameinfo->chromaImg.create (input[0]->props.height / 2,
          input[0]->props.stride / 2, CV_16UC1);
      frameinfo->chromaImg.data = (unsigned char *) chromaBuf;
    } else if (frameinfo->inframe->props.fmt == VVAS_VFMT_BGR8) {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "Input frame is in BGR format\n");
      frameinfo->image.create (input[0]->props.height,
          input[0]->props.stride / 3, CV_8UC3);
      frameinfo->image.data = (unsigned char *) indata;
    } else {
      LOG_MESSAGE (LOG_LEVEL_WARNING, "Unsupported color format\n");
      return 0;
    }


    if (infer_meta != NULL) {
    /* Print the entire prediction tree */
    pstr = gst_inference_prediction_to_string (infer_meta->prediction);
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "Prediction tree: \n%s", pstr);
    free (pstr);

    g_node_traverse (infer_meta->prediction->predictions, G_PRE_ORDER,
        G_TRAVERSE_ALL, -1, overlay_node_foreach, kpriv);
    }

    fps_overlay(kpriv);
    return 0;
  }


  int32_t xlnx_kernel_done (VVASKernel * handle)
  {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "enter");
    return 0;
  }
}