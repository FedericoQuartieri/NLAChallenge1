#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>
// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

typedef Eigen::Triplet<double> T;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    const char* input_image_path = argv[1];

    // Load the image using stb_image
    int cols, rows, channels;
    // for greyscale images force to load only one channel
    unsigned char* image_data;
    image_data = stbi_load(input_image_path, &cols, &rows, &channels, 1);
    if (!image_data) {
        std::cerr << "Error: Could not load image " << input_image_path << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << rows << "x" << cols << " with " << channels << " channels." << std::endl;
    MatrixXd inputMatrix(rows,cols);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            int index = (i * cols + j) * 1;
            inputMatrix(i,j) = static_cast<double>(image_data[index]) / 255.0;
        }
    }
    stbi_image_free(image_data);

  Eigen::MatrixXd randomMatrix = Eigen::MatrixXd::Random(rows, cols);
  randomMatrix = 50*randomMatrix;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> noised(rows, cols);//you have to specify that the matrix is rowmajor!
  
  // Fill the matrices with image data
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int index = (i * cols + j) * channels;  // 1 channel (Greyscale) 3 channels (RGB)
      
      noised(i, j) = (static_cast<double>(image_data[index]) + randomMatrix(i, j)) / 255.0;
      if (noised(i,j) >= 1) noised(i, j) = 1;
      if (noised(i,j) <= 0) noised(i, j) = 0;

    }
  }

  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> output_image(rows, cols);
  // Use Eigen's unaryExpr to map the inputMatrixscale values (0.0 to 1.0) to 0 to 255
    output_image = noised.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val * 255.0);
  });
    // Save the image using stb_image_write
  const std::string output_image_path1 = "noised_task2.png";
  if (stbi_write_png(output_image_path1.c_str(), cols, rows, 1, output_image.data(), cols) == 0) {
    std::cerr << "Error: Could not save inputMatrixscale image" << std::endl;

    return 1;
  }


  //----------------------here starts task 3-----------------------------------

  Eigen::VectorXd original = Eigen::Map<Eigen::VectorXd>(inputMatrix.data(), inputMatrix.size());
  Eigen::VectorXd noisedVector = Eigen::Map<Eigen::VectorXd>(noised.data(), noised.size());
  std::cout <<"The size of the vectors are:" << original.size() << ", " << noisedVector.size() << std::endl;

  std::cout <<"The norm of the original matrix flattened to a vector is: " << original.norm() << std::endl;

  //Eigen::VectorXd diff = original - noisedVector;
  SparseMatrix<double> A1(rows*cols, rows*cols);

  std::cout << A1.size() << std::endl;
  int c = 0;

  std::vector<T> tripletList;
  tripletList.reserve(782086);
  for (int i = 0; i < rows; i++) {
  for (int j = 0; j < cols; j++) {
    for (int k = i - 1; k <= i + 1; k++) {
      for (int l = j - 1; l <= j + 1; l++) {
        if (k >= 0 && k < rows && l >= 0 && l < cols) {
          tripletList.emplace_back(T(i * cols + j, k * cols + l, 1.0 / 9));
          c++;
        }
      }
    }
  }
}

  A1.setFromTriplets(tripletList.begin(), tripletList.end());
  
  //std::cout << A1 << std::endl;
  std::cout << "number of non zero values= " << c << std::endl;

  //-------------------------------Here starts task 5----------------------------------------

  //std::cout << A1;

  VectorXd ris1 = A1*noisedVector;
  std::cout << ris1.size() << std::endl;
  //MatrixXd smoothed(341,256);
  
  //std::cout << ris1;

  //Eigen::MatrixXd smoothed = ris1.reshaped(rows, cols);
  
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> smoothed = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(ris1.data(), rows, cols);

// Apply clipping to ensure values stay between 0 and 1
smoothed = smoothed.unaryExpr([](double val) -> double {
    return std::min(1.0, std::max(0.0, val));  // Clip values between 0 and 1
});

// Convert the clipped matrix to unsigned char for saving the image
Matrix<unsigned char, Dynamic, Dynamic, RowMajor> output_image_smoothed(rows, cols);
output_image_smoothed = smoothed.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val * 255.0);
});

// Save the image using stb_image_write
const std::string output_image_path2 = "noised_task5.png";
if (stbi_write_png(output_image_path2.c_str(), cols, rows, 1, output_image_smoothed.data(), cols) == 0) {
    std::cerr << "Error: Could not save smoothed image" << std::endl;
    return 1;
}

  //std::cout << smoothed ;

  //std::cout << noised << st;

  std::cout << smoothed.cols() << std::endl;

  return 0;
  }