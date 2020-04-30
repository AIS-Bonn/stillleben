// stillleben differentiation module Python bridge
// Author: Arul Periyasamy <arul.periyasamy@ais.uni-bonn.de>

#include <torch/extension.h>

#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "diff.h"

namespace
{
    torch::Tensor generateSobelValidMask(torch::Tensor& instance_indices, torch::Tensor depth_image)
    {
        if(instance_indices.dim() != 2 || depth_image.dim() != 2)
            throw std::invalid_argument{"input tensors should be two-dimensional"};
        if(instance_indices.size(0) != depth_image.size(0) || instance_indices.size(1) != depth_image.size(1))
            throw std::invalid_argument{"instance_indices and depth_image should be of same height and width"};

        int H = instance_indices.size(0);
        int W = instance_indices.size(1);

        if(!depth_image.is_contiguous())
            depth_image = depth_image.contiguous();

        auto device = instance_indices.device();
        auto device_type = instance_indices.device().type();

        // all pixels are valid to start with
        // invalidate the occluded pixels
        torch::Tensor valid_mask = torch::ones({H,  W}, torch::TensorOptions().dtype(torch::kBool).device(device));
        if(device_type == torch::kCUDA)
        {
            diff::generateSobelValidMaskCuda(instance_indices, depth_image, valid_mask);
        }
        else
        {
            auto indices_accessor = instance_indices.accessor<int16_t, 2>();
            auto depth_accessor = depth_image.accessor<float, 2>();
            auto mask_accessor = valid_mask.accessor<bool, 2>();
            for(int h=1; h<H-1; ++h)
            {
                for(int w=1; w<W-1; ++w)
                {
                    // background pixels are not interesting
                    if(indices_accessor[h][w] == 0)
                        continue;

                    auto current_index = indices_accessor[h][w];
                    auto current_depth = depth_accessor[h][w];

                    for(int x=-1; x<=1; ++x)
                    {
                        for(int y=-1; y<=1; ++y)
                        {
                            if((indices_accessor[h + x][w + y] != current_index) && (indices_accessor[h + x][w + y] != 0) && (depth_accessor[h + x][w + y] < current_depth))
                            {
                                // presence of occlusion
                                // neighboring pixel is in foreground
                                mask_accessor[h][w] = 0;
                            }
                        }
                    }
                } // for W
            } // for H
        } // else CPU

        return valid_mask;
    }

    std::tuple<torch::Tensor, torch::Tensor> dilateObjectMask(torch::Tensor& object_mask, torch::Tensor& sobel_valid_mask, torch::Tensor& coordinates)
    {
        if(object_mask.dim() != 2 || sobel_valid_mask.dim() != 2)
            throw std::invalid_argument{"object_mask &  sobel_valid_mask should be two-dimensional"};
        if(coordinates.dim() != 3)
            throw std::invalid_argument{"coordinates should be three-dimensional"};
        if( object_mask.size(0) != sobel_valid_mask.size(0) || object_mask.size(1) != sobel_valid_mask.size(1)
            || object_mask.size(0) != coordinates.size(0) || object_mask.size(1) != coordinates.size(1) )
            throw std::invalid_argument{"object_mask, sobel_valid_mask, and coordinates should be of same height and width"};

        int H = object_mask.size(0);
        int W = object_mask.size(1);

        auto device = object_mask.device();
        auto device_type = device.type();

        if(!coordinates.is_contiguous())
            coordinates = coordinates.contiguous();

        torch::Tensor dilated_mask = torch::zeros({H,  W}, torch::TensorOptions().dtype(torch::kBool).device(device)) ;
        torch::Tensor dilated_coordinates = torch::zeros({H,  W, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

        if(device_type == torch::kCUDA)
        {
            diff::dilateObjectMaskCuda(object_mask, sobel_valid_mask, coordinates, dilated_mask, dilated_coordinates);
        }
        else
        {
            auto object_mask_accessor = object_mask.accessor<bool, 2>();
            auto sobel_valid_accessor = sobel_valid_mask.accessor<bool, 2>();
            auto dilated_mask_accessor = dilated_mask.accessor<bool, 2>();
            auto coordinates_accessor = coordinates.accessor<float, 3>();
            auto dilated_coordinates_accessor = dilated_coordinates.accessor<float, 3>();

//                 #pragma omp parallel for
            for(int h=1; h<H-1; ++h)
            {
                for(int w=1; w<W-1; ++w)
                {
                    // copy current label to start with.
                    dilated_mask_accessor[h][w] = object_mask_accessor[h][w];
                    dilated_coordinates_accessor[h][w][0] = coordinates_accessor[h][w][0];
                    dilated_coordinates_accessor[h][w][1] = coordinates_accessor[h][w][1];
                    dilated_coordinates_accessor[h][w][2] = coordinates_accessor[h][w][2];

                    // do we have to do anything for this pixel?
                    if(object_mask_accessor[h][w] != 0)
                        continue;

                    // check if none of the pixels is invalid
                    bool all_valid = true;
                    bool all_background = true;
                    float cx=0,cy=0,cz=0;

                    for(int x=-1; x<=1; ++x)
                    {
                        for(int y=-1; y<=1; ++y)
                        {
                            if(object_mask_accessor[h + x][w + y] != 0)
                            {
                                all_background = false;
                                cx = coordinates_accessor[h + x][w + y][0];
                                cy = coordinates_accessor[h + x][w + y][1];
                                cz = coordinates_accessor[h + x][w + y][2];
                            }

                            if(sobel_valid_accessor[h + x][w + y] == 0)
                            {
                                all_valid = false;
                                break;
                            }
                        }
                    }

                    if(all_background || !all_valid)
                        continue;

                    dilated_mask_accessor[h][w] = 1;
                    dilated_coordinates_accessor[h][w][0] = cx;
                    dilated_coordinates_accessor[h][w][1] = cy;
                    dilated_coordinates_accessor[h][w][2] = cz;
                } // for W
            } // for H
        } // else CPU

        return std::make_tuple(dilated_mask, dilated_coordinates);
    }
}

PYBIND11_MODULE(libstillleben_diff_python, m)
{
    // generate sobel valid mask for gradient for occlusion case
    m.def("generate_sobel_valid_mask", generateSobelValidMask,
        R"EOS(
            Generate mask of valid pixels.

            :param instance_indices: HxW short tensor with instance indices
            :param depth_image: HxW float tensor with depth
            :return: HxW bool tensor

            The returned mask is unset iff the pixel is close to an occluder
            (i.e. there is a neighboring pixel of another object that is closer).
        )EOS");

    // dilate object mask
    m.def("dilate_object_mask", dilateObjectMask,
        R"EOS(
            Dilate object mask.
        )EOS");
}
