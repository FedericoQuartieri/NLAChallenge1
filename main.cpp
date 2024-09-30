#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstdlib>

// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    const char* input_image_path = argv[1];

    // Load the image using stb_image
    int width, height, channels;
    // for greyscale images force to load only one channel
    unsigned char* image_data;
    image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
    if (!image_data) {
        std::cerr << "Error: Could not load image " << input_image_path << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;
    MatrixXd inputMatrix(height,width);
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            int index = (i * width + j) * 1;
            inputMatrix(i,j) = static_cast<double>(image_data[index]) / 255.0;
        }
    }
    stbi_image_free(image_data);

  Eigen::MatrixXd randomMatrix = Eigen::MatrixXd::Random(height, width);
  randomMatrix = 50*randomMatrix;
  MatrixXd noised(height, width);
  
  // Fill the matrices with image data
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int index = (i * width + j) * channels;  // 1 channel (Greyscale) 3 channels (RGB)
      
      noised(i, j) = (static_cast<double>(image_data[index]) + randomMatrix(i, j)) / 255.0;
      if (noised(i,j) >= 1) noised(i, j) = 1;
      if (noised(i,j) <= 0) noised(i, j) = 0;

    }
  }

  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> output_image(width, height);
  // Use Eigen's unaryExpr to map the inputMatrixscale values (0.0 to 1.0) to 0 to 255
    output_image = noised.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val * 255.0);
  });
    // Save the image using stb_image_write
  const std::string output_image_path1 = "noised_task2.png";
  if (stbi_write_png(output_image_path1.c_str(), width, height, 1,
                     output_image.data(), width) == 0) {
    std::cerr << "Error: Could not save inputMatrixscale image" << std::endl;

    return 1;
  }


  //----------------------here starts task 3-----------------------------------

  Eigen::VectorXd original = Eigen::Map<Eigen::VectorXd>(inputMatrix.data(), inputMatrix.size());
  Eigen::VectorXd noisedVector = Eigen::Map<Eigen::VectorXd>(noised.data(), noised.size());
  std::cout <<"The size of the vectors are:" << original.size() << ", " << noisedVector.size() << std::endl;

  std::cout <<"The norm of the original matrix flattened to a vector is: " << original.norm() << std::endl;

  Eigen::VectorXd diff = original - noisedVector;


  SparseMatrix<double> A1(height*width, height*width);
  
  

  std::cout << A1.size() << std::endl;
  int c = 0;

  for (int i = 0; i < height; i++){
    for (int j = 0; j < width; j++){
      for (int k = i-1; k <= i + 1; k++){
        for (int l = j-1; l<= j + 1 ; l++){
          if(k >= 0 && k < height && l >= 0 && l < width){
            //std::cout << "c = " << c << std::endl;
            A1.coeffRef(i*width + j, k*width + l) = 1.0/9;
            //A1.insert(i*width + j, k*width + l) = 1.0/9;
            c++;
          }
        }
      }
      // single row of A1 completed
    }
    std::cout << "i = " << i << std::endl;
  }
  std::cout << "ehijnsk";
  //A1.makeCompressed();
  //std::cout << A1 << std::endl;
  std::cout << c << std::endl;

  return 0;
}