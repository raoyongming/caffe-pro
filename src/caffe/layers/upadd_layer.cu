#include <cfloat>
#include <vector>

#include "caffe/layers/upadd_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void UpaddForward(const int nthreads, const Dtype* bottom_data_a,
    const Dtype* bottom_data_b, const int channels, const int height_a, const int width_a, 
    const int height_b, const int width_b, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % width_a;
    int h = (index / width_a) % height_a;
    int c = (index / width_a/ height_a) % channels;
    int n = (index / width_a/ height_a / channels);

    w = (w/2 >= width_b) ? (width_b - 1): (w/2);
    h = (h/2 >= height_b)? (height_b - 1): (h/2);

    int idx = w + h*width_b + c*width_b*height_b + n*width_b*height_b*channels;

    top_data[index] = bottom_data_a[index] + bottom_data_b[idx];
  }
}

template <typename Dtype>
void UpaddLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data_a = bottom[0]->gpu_data();
  const Dtype* bottom_data_b = bottom[0]->gpu_data();

  const int height_a = bottom[0]->shape(2);
  const int width_a = bottom[0]->shape(3);
  const int height_b = bottom[1]->shape(2);
  const int width_b = bottom[1]->shape(3);

  const int channels = bottom[0]->shape(1);
  UpaddForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data_a, bottom_data_b, channels, height_a, width_a, height_b, width_b, top_data);
}

template <typename Dtype>
__global__ void UpaddBackward(const int nthreads, const Dtype* top_diff,
    const int channels, const int height_a, const int width_a, 
    const int height_b, const int width_b, Dtype* bottom_diff_b){
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % width_a;
    int h = (index / width_a) % height_a;
    int c = (index / width_a/ height_a) % channels;
    int n = (index / width_a/ height_a / channels);

    w = (w*2 >= width_a)? (width_a - 1): (w*2);
    h = (h*2 >= height_a)? (height_a - 1): (h*2);


    int idx = w + h*width_a + c*width_a*height_a + n*width_a*height_a*channels;

    bottom_diff_b[index] = top_diff[idx];
  }

}

template <typename Dtype>
void UpaddLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[1]->count();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff_a = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_diff_b = bottom[0]->mutable_gpu_diff();

  const int height_a = bottom[0]->shape(2);
  const int width_a = bottom[0]->shape(3);
  const int height_b = bottom[1]->shape(2);
  const int width_b = bottom[1]->shape(3);

  const int channels = bottom[0]->shape(1);



  UpaddBackward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, channels, height_a, width_a, height_b, width_b, bottom_diff_b);

  caffe_copy(bottom[0]->count(), top_diff, bottom_diff_a);

}

INSTANTIATE_LAYER_GPU_FUNCS(UpaddLayer);

}  // namespace caffe
