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
using namespace std;

typedef Eigen::Triplet<double> T;


//Matrix<unsigned char, Dynamic, Dynamic, RowMajor>
int saveImage(int rows, int cols, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> image, string output_image_path){
  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> output_image(rows, cols);
    // Use Eigen's unaryExpr to map the inputMatrixscale values (0.0 to 1.0) to 0 to 255
    output_image = image.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val * 255.0);
  });

  // Save the image using stb_image_write
  if (stbi_write_png(output_image_path.c_str(), cols, rows, 1, output_image.data(), cols) == 0) {
    cerr << "Error: Could not save inputMatrixscale image" << endl;
    return 1;
  }
  return 0;
}


bool isSymmetricPositiveDefinite(const SparseMatrix<double>& matrix) {
    // Verifica se la matrice è quadrata
    if (matrix.rows() != matrix.cols()) {
        cerr << "La matrice non è quadrata!" << endl;
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

Eigen::VectorXd loadVectorFromMTX(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Cannot open file.");
    }

    std::string line;
    // Salta i commenti e l'intestazione
    while (std::getline(infile, line)) {
        if (line[0] != '%') {
            break;
        }
    }

    // Legge la dimensione del vettore (e altre informazioni inutilizzate)
    int rows;
    std::istringstream iss(line);
    iss >> rows;

    // Assicuriamoci che il file descriva un vettore colonna
    //if (cols != 1) {
      //  throw std::runtime_error("The file does not contain a column vector.");
    //}

    // Creiamo il vettore Eigen::VectorXd della dimensione corretta
    Eigen::VectorXd vec(rows);

    // Legge i valori dal file .mtx
    int index;
    double value;
    for (int i = 0; i < rows; ++i) {
        infile >> index >> value;
        vec(index - 1) = value;  // Gli indici in Matrix Market partono da 1, quindi dobbiamo sottrarre 1
    }

    infile.close();
    return vec;
}


void exportToMatrixMarket(const Eigen::SparseMatrix<double>& mat, const std::string& filename) {
    std::ofstream file(filename);

    // Check if the file opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write the Matrix Market header
    file << "%%MatrixMarket matrix coordinate real general\n";
    file << mat.rows() << " " << mat.cols() << " " << mat.nonZeros() << "\n";

    // Loop through non-zero elements of the sparse matrix
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
            // MatrixMarket format uses 1-based indexing, so add 1 to row and col indices
            file << (it.row() + 1) << " " << (it.col() + 1) << " " << it.value() << "\n";
        }
    }

    // Close the file
    file.close();
}
void exportVectorToMatrixMarket(const Eigen::VectorXd& vec, const std::string& filename) {
    std::ofstream file(filename);

    // Check if the file opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write the Matrix Market header
    file << "%%MatrixMarket vector array real\n";
    file << vec.size() << "\n";  // Write the number of values

    // Write the vector values
    for (int i = 0; i < vec.size(); ++i) {
        file << vec(i) << "\n";  // Write only the values, no indexing
    }

    // Close the file
    file.close();
}


int main(int argc, char* argv[]) {
  
  //---------------------------setup--------------------------

  if (argc < 2) {
      cerr << "Usage: " << argv[0] << " <image_path>" << endl;
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
      cerr << "Error: Could not load image " << input_image_path << endl;
      return 1;
  }

  cout << "Image loaded: " << rows << "x" << cols << " with " << channels << " channels." << endl;
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



  saveImage(rows, cols, noised, "noised_task2.png");



  //----------------------here starts task 3-----------------------------------
  //originalVector
  //noisedVector


  Eigen::VectorXd originalVector = Eigen::Map<Eigen::VectorXd>(inputMatrix.data(), inputMatrix.size());
  Eigen::VectorXd noisedVector = Eigen::Map<Eigen::VectorXd>(noised.data(), noised.size());
  cout <<"The size of the vectors are:" << originalVector.size() << ", " << noisedVector.size() << endl;

  cout <<"The norm of the originalVector matrix flattened to a vector is: " << originalVector.norm() << endl;


  //----------------------here starts task 4 (A1)-----------------------------------
  //A1


  //Eigen::VectorXd diff = originalVector - noisedVector;
  SparseMatrix<double> A1(rows*cols, rows*cols);

  cout << A1.size() << endl;
  int c = 0;

  vector<T> tripletList;
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

  //cout << A1 << endl;
  cout << "number of non zero values= " << c << endl;

  //-------------------------------Here starts task 5 (apply A1)----------------------------------------
  //smoothed


  VectorXd ris1 = A1*noisedVector;
  cout << ris1.size() << endl;


  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> smoothed = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(ris1.data(), rows, cols);

  // Apply clipping to ensure values stay between 0 and 1
  smoothed = smoothed.unaryExpr([](double val) -> double {
    return min(1.0, max(0.0, val));  // Clip values between 0 and 1
  });

  saveImage(rows, cols, smoothed, "smoothed_task5.png");



  //-------------------------Here starts task 6 (A2)--------------------------
  //A2

  int nnz=0;
  vector<T> tripletsList;
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
  cout << "number of non zero values= " << nnz << endl;

  if (A2.isApprox(A2.transpose())) {
        cout << "The matrix A2 is symmetric." << endl;
    } else {
        cout << "The matrix A2 is NOT SYMMETRIC" << endl;
    }


  //-------------------------Here starts task 7 (apply A2)--------------------------
  //sharpened

  VectorXd ris2 = A2*originalVector;
  cout << ris2.size() << endl;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> sharpened = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(ris2.data(), rows, cols);

  // Apply clipping to ensure values stay between 0 and 1
  sharpened = sharpened.unaryExpr([](double val) -> double {
    return min(1.0, max(0.0, val));  // Clip values between 0 and 1
  });

  saveImage(rows, cols, sharpened, "sharpened_task7.png");



  //-------------------------Here starts task 8--------------------------
  
  exportToMatrixMarket(A2,"matrix_output.mtx");
  exportVectorToMatrixMarket(noisedVector*255,"vector_output.mtx");
  //tol=1e-9;
 


  //-------------------------Here starts task 9--------------------------

  VectorXd task9Vector = loadVectorFromMTX("sol.mtx");
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> task9 = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(task9Vector.data(), rows, cols);

  // Apply clipping to ensure values stay between 0 and 1
  task9 = task9.unaryExpr([](double val) -> double {
    return min(1.0, max(0.0, val));  // Clip values between 0 and 1
  });

  saveImage(rows, cols, task9, "task9.png");






  //-------------------------Here starts task 10 (A3)--------------------------

  int nonzero=0;
  vector<T> triplets;
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
  cout << "number of non zero values= " << nonzero << endl;

  if (A3.isApprox(A3.transpose())) {
      cout << "The matrix A3 is symmetric." << endl;
    } else {
      cout << "The matrix A3 is NOT SYMMETRIC" << endl;
  }




  //-------------------------Here starts task 11 (apply A3)--------------------------
  //edge detection

  VectorXd ris3 = A3*originalVector;
  cout << ris3.size() << endl;


  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> detected = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(ris3.data(), rows, cols);

  // Apply clipping to ensure values stay between 0 and 1
  detected = detected.unaryExpr([](double val) -> double {
    return min(1.0, max(0.0, val));  // Clip values between 0 and 1
  });

  saveImage(rows, cols, detected, "edge_detection_task11.png");



  //-------------------------Here starts task 12()--------------------------


  SparseMatrix<double> I(rows*cols, rows*cols);
  vector<T> identity;
  for(int i = 0;i<rows*cols;i++){
      identity.emplace_back(i,i,1);
  }
  I.setFromTriplets(identity.begin(),identity.end());
  SparseMatrix<double> A4(rows*cols, rows*cols);
  A4 = I + A3;

  if (isSymmetricPositiveDefinite(A4)) {
        cout << "La matrice è simmetrica e definita positiva." << endl;
    } else {
        cout << "La matrice non è simmetrica o non è definita positiva." << endl;
    }
    
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cg;
  cg.compute(A4);
  VectorXd w=noisedVector;
  double tol = 1e-10;
  cg.setTolerance(tol);
  VectorXd y=cg.solve(w);
  
  cout << "Iterations:" << cg.iterations() << endl;
  cout << "Residual:" << cg.error() << endl;


  //-------------------------Here starts task 13 -------------------------
  
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> task13 = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(y.data(), rows, cols);

  // Apply clipping to ensure values stay between 0 and 1
  task13 = task13.unaryExpr([](double val) -> double {
    return min(1.0, max(0.0, val));  // Clip values between 0 and 1
  });

  saveImage(rows, cols, task13, "task13.png");



  //-------------------------Here starts task 14 -------------------------







  //-------------------------Here starts task 15--------------------------



















    
  return 0;
}