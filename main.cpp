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



// int saveImage(int rows, int cols, Matrix<unsigned char, Dynamic, Dynamic, RowMajor> image){
//   Matrix<unsigned char, Dynamic, Dynamic, RowMajor> output_image(rows, cols);
//     // Use Eigen's unaryExpr to map the inputMatrixscale values (0.0 to 1.0) to 0 to 255
//     output_image = image.unaryExpr([](double val) -> unsigned char {
//     return static_cast<unsigned char>(val * 255.0);
//   });
//   // Save the image using stb_image_write
//   const std::string output_image_path1 = "noised_task2.png";
//   if (stbi_write_png(output_image_path1.c_str(), cols, rows, 1, output_image.data(), cols) == 0) {
//     std::cerr << "Error: Could not save inputMatrixscale image" << std::endl;
//     return 1;
//   }
// }
bool isSymmetricPositiveDefinite(const SparseMatrix<double>& matrix) {
    // Verifica se la matrice è quadrata
    if (matrix.rows() != matrix.cols()) {
        std::cerr << "La matrice non è quadrata!" << std::endl;
        return false;
    }

    // Prova la fattorizzazione di Cholesky
    SimplicialLLT<SparseMatrix<double>> cholesky;
    cholesky.compute(matrix);

    // Se ci sono errori nella fattorizzazione, la matrice non è definita positiva
    if (cholesky.info() != Success) {
        return false;
    }

    return true;
}



int main(int argc, char* argv[]) {
  
  //---------------------------setup--------------------------

  if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
      return 1;
  }


  //---------------------------load image (TASK1)--------------------------
  //InputMatrix


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
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> inputMatrix(rows, cols);
  for(int i=0;i<rows;i++){
      for(int j=0;j<cols;j++){
          int index = (i * cols + j) * 1;
          inputMatrix(i,j) = static_cast<double>(image_data[index]) / 255.0;
      }
  }
  stbi_image_free(image_data);
  
  
  //-----------------introduce noise (TASK2)-------------------------
  //noised


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

  //saveImage(rows, cols, noised);

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

  Eigen::VectorXd originalVector = Eigen::Map<Eigen::VectorXd>(inputMatrix.data(), inputMatrix.size());
  Eigen::VectorXd noisedVector = Eigen::Map<Eigen::VectorXd>(noised.data(), noised.size());
  std::cout <<"The size of the vectors are:" << originalVector.size() << ", " << noisedVector.size() << std::endl;

  std::cout <<"The norm of the originalVector matrix flattened to a vector is: " << originalVector.norm() << std::endl;

  //Eigen::VectorXd diff = originalVector - noisedVector;
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


  //-------------------------Here starts task 6--------------------------

  int nnz=0;
  std::vector<T> tripletsList;
  tripletList.reserve(782086);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      for (int k = i - 1; k <= i + 1; k++) {
        for (int l = j - 1; l <= j + 1; l++) {
          if (k >= 0 && k < rows && l >= 0 && l < cols) {
            if(k==i&&l==j){
              tripletsList.emplace_back(T(i * cols + j, k * cols + l, 9.0));
              nnz++;
            }
            else if(k==i){
              if(l==j-1){
                tripletsList.emplace_back(T(i * cols + j, k * cols + l, -1.0));
                nnz++;
              }
              else if(l==j+1){
                tripletsList.emplace_back(T(i * cols + j, k * cols + l, -3.0));
                nnz++;
              }
              
            }
            else if(l==j){
              if(k==i-1){
                tripletsList.emplace_back(T(i * cols + j, k * cols + l, -3.0));
                nnz++;
              }
              else if(k==i+1){
                tripletsList.emplace_back(T(i * cols + j, k * cols + l, -1.0));
                nnz++;
              }
            }
            
          }
        }
      }
    }
  }
  SparseMatrix<double> A2(rows*cols, rows*cols);
  A2.setFromTriplets(tripletsList.begin(), tripletsList.end());
  std::cout << "number of non zero values= " << nnz << std::endl;

  if (A2.isApprox(A2.transpose())) {
        std::cout << "The matrix is symmetric." << std::endl;
    } else {
        std::cout << "Die alone you can't code." << std::endl;
    }


  //-------------------------Here starts task 7--------------------------

  //std::cout << A1;

  VectorXd ris2 = A2*originalVector;
  std::cout << ris2.size() << std::endl;
  //MatrixXd smoothed(341,256);

  //std::cout << ris1;

  //Eigen::MatrixXd smoothed = ris1.reshaped(rows, cols);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> sharpened = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(ris2.data(), rows, cols);

  // Apply clipping to ensure values stay between 0 and 1
  sharpened = sharpened.unaryExpr([](double val) -> double {
    return std::min(1.0, std::max(0.0, val));  // Clip values between 0 and 1
  });

  // Convert the clipped matrix to unsigned char for saving the image
  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> output_image_sharpened(rows, cols);
  output_image_sharpened = sharpened.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val * 255.0);
  });

  // Save the image using stb_image_write
  const std::string output_image_path3 = "task7.png";
  if (stbi_write_png(output_image_path3.c_str(), cols, rows, 1, output_image_sharpened.data(), cols) == 0) {
    std::cerr << "Error: Could not save smoothed image" << std::endl;
    return 1;
  }

  //std::cout << smoothed ;

  //std::cout << noised << st;

  //std::cout << smoothed.cols() << std::endl;





  //-------------------------Here starts task 8--------------------------





  //-------------------------Here starts task 9--------------------------






  //-------------------------Here starts task 10--------------------------

  int nonzero=0;
  std::vector<T> triplets;
  triplets.reserve(782086);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      for (int k = i - 1; k <= i + 1; k++) {
        for (int l = j - 1; l <= j + 1; l++) {
          if (k >= 0 && k < rows && l >= 0 && l < cols) {
            if(k==i&&l==j){
              triplets.emplace_back(T(i * cols + j, k * cols + l, 4));
              nonzero++;
            }
            else if(k==i){
              if(l==j-1){
                triplets.emplace_back(T(i * cols + j, k * cols + l, -1.0));
                nonzero++;
              }
              else if(l==j+1){
                triplets.emplace_back(T(i * cols + j, k * cols + l, -1.0));
                nonzero++;
              }
              
            }
            else if(l==j){
              if(k==i-1){
                triplets.emplace_back(T(i * cols + j, k * cols + l, -1.0));
                nonzero++;
              }
              else if(k==i+1){
                triplets.emplace_back(T(i * cols + j, k * cols + l, -1.0));
                nonzero++;
              }
            }
            
          }
        }
      }
    }
  }
  SparseMatrix<double> A3(rows*cols, rows*cols);
  A3.setFromTriplets(triplets.begin(), triplets.end());
  std::cout << "number of non zero values= " << nonzero << std::endl;

  if (A3.isApprox(A3.transpose())) {
        std::cout << "The matrix A3 is symmetric." << std::endl;
    } else {
        std::cout << "Die alone you can't code." << std::endl;
    }






  //-------------------------Here starts task 11--------------------------
  //std::cout << A1;

  VectorXd ris3 = A3*originalVector;
  std::cout << ris3.size() << std::endl;
  //MatrixXd smoothed(341,256);

  //std::cout << ris1;

  //Eigen::MatrixXd smoothed = ris1.reshaped(rows, cols);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> detected = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(ris3.data(), rows, cols);

  // Apply clipping to ensure values stay between 0 and 1
  detected = detected.unaryExpr([](double val) -> double {
    return std::min(1.0, std::max(0.0, val));  // Clip values between 0 and 1
  });

  // Convert the clipped matrix to unsigned char for saving the image
  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> output_image_detected(rows, cols);
  output_image_detected = detected.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val * 255.0);
  });

  // Save the image using stb_image_write
  const std::string output_image_path4 = "task11.png";
  if (stbi_write_png(output_image_path4.c_str(), cols, rows, 1, output_image_detected.data(), cols) == 0) {
    std::cerr << "Error: Could not save smoothed image" << std::endl;
    return 1;
  }

  //std::cout << smoothed ;

  //std::cout << noised << st;

  //std::cout << smoothed.cols() << std::endl;




  //-------------------------Here starts task 12--------------------------
  SparseMatrix<double> I(rows*cols, rows*cols);
  std::vector<T> identity;
  for(int i = 0;i<rows*cols;i++){
      identity.emplace_back(i,i,1);
  }
  I.setFromTriplets(identity.begin(),identity.end());
  SparseMatrix<double> A4(rows*cols, rows*cols);
  A4 = I + A3;

  if (isSymmetricPositiveDefinite(A4)) {
        std::cout << "La matrice è simmetrica e definita positiva." << std::endl;
    } else {
        std::cout << "La matrice non è simmetrica o non è definita positiva." << std::endl;
    }
  

  //-------------------------Here starts task 13 -------------------------






  //-------------------------Here starts task 14 -------------------------







  //-------------------------Here starts task 15--------------------------



















    
  return 0;
}