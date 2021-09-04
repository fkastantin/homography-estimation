#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;



// This function is used for data normalization, data normalization is needed due to numerical computation (calculate eigenvectors)
// translation: origin should be at the center of gravity
// scale: spread should be set to sqrt(2)
void normalizePoints(
	vector<Point2f> &points_, // input, original points
	vector<Point2f> &normalized_points_, // output, normalized points
	Mat &transformation_matrix_ // output, transformation matrix from original to normalized coordinates
) {
	int n = points_.size();

	// Calc the masspoint for the all points
	Point2f masspoint(0, 0);
	for (int i = 0; i < n; i++) {
		masspoint += points_[i];
	}
	masspoint /= n;

	// Translate all points so that the origin is the center of gravity
	for (int i = 0; i < n; i++) {
		Point2f p = points_[i] - masspoint;
		normalized_points_.push_back(p);
	}

	// Calculate points' average distnace from the origin
	double average_distance = 0.0;
	for (int i = 0; i < n; i++) {
		average_distance += norm(normalized_points_[i]);
	}
	average_distance /= n;

	// sclae so that spread should be set to sqrt(2)
	const double ratio = sqrt(2) / average_distance;
	for (int i = 0; i < n; i++) {
		normalized_points_[i] *= ratio;
	}

	transformation_matrix_ = Mat::eye(3, 3, CV_64F);
	transformation_matrix_.at<double>(0, 0) = ratio;
	transformation_matrix_.at<double>(1, 1) = ratio;
	transformation_matrix_.at<double>(0, 2) = -masspoint.x * ratio;
	transformation_matrix_.at<double>(1, 2) = -masspoint.y * ratio;
}


void calcHomographyMatrix(
	vector<Point2f> &points_1_, // input, all correspondences
	vector<Point2f> &points_2_, 
	vector<int> &indices, // input,  indiceis of points to rely on when calculating the homography matrix
	Mat &H // output, homography estimation matrix
) {
	int number_of_points = indices.size();

	Mat A(2 * number_of_points, 9, CV_32F);

	for (int i = 0; i < number_of_points; i++) {
		float u1 = points_1_[indices[i]].x;
		float v1 = points_1_[indices[i]].y;

		float u2 = points_2_[indices[i]].x;
		float v2 = points_2_[indices[i]].y;

		A.at<float>(2 * i, 0) = u1;
		A.at<float>(2 * i, 1) = v1;
		A.at<float>(2 * i, 2) = 1.0f;
		A.at<float>(2 * i, 3) = 0.0f;
		A.at<float>(2 * i, 4) = 0.0f;
		A.at<float>(2 * i, 5) = 0.0f;
		A.at<float>(2 * i, 6) = -u2 * u1;
		A.at<float>(2 * i, 7) = -u2 * v1;
		A.at<float>(2 * i, 8) = -u2;

		A.at<float>(2 * i + 1, 0) = 0.0f;
		A.at<float>(2 * i + 1, 1) = 0.0f;
		A.at<float>(2 * i + 1, 2) = 0.0f;
		A.at<float>(2 * i + 1, 3) = u1;
		A.at<float>(2 * i + 1, 4) = v1;
		A.at<float>(2 * i + 1, 5) = 1.0f;
		A.at<float>(2 * i + 1, 6) = -v2 * u1;
		A.at<float>(2 * i + 1, 7) = -v2 * v1;
		A.at<float>(2 * i + 1, 8) = -v2;

	}

	Mat eVecs(9, 9, CV_32F), eVals(9, 9, CV_32F);

	eigen(A.t() * A, eVals, eVecs);

	for (int i = 0; i < 9; i++) 
		H.at<double>(i / 3, i % 3) = eVecs.at<float>(8, i);
}

// Distance between 2 points
double calcDistance(Point2f &point1_, Point2f &point2_) {
	double dx = point1_.x - point2_.x;
	double dy = point2_.y - point2_.y;

	return sqrt(dx * dx + dy * dy);
}


int getIterationNumber(int point_number_,
	int inlier_number_,
	int sample_size_,
	double confidence_)
{
	const double inlier_ratio =
		static_cast<double>(inlier_number_) / point_number_;

	static const double log1 = log(1.0 - confidence_);
	const double log2 = log(1.0 - pow(inlier_ratio, sample_size_));

	const int k = log1 / log2;
	if (k < 0)
		return std::numeric_limits<int>::max();
	return k;
}

// Find best homography estimation using RANSAC
void ransacBestHomographyEstimation(
	vector<Point2f> &points_1_, // input, matched corresponding points
	vector<Point2f> &points_2_,
	vector<Point2f> &normalized_points_1_,
	vector<Point2f> &normalized_points_2_,
	Mat &T_1_,
	Mat &T_2_,
	const int max_iterations_, // input, ransac iterations number
	const double ransac_threshold_, // input, ransac threshold
	const double ransac_confidence_,
	Mat &bestH_  // output, best homgraphy estimation
) {
	int best_inliers = -1;
	int max_iterations_with_confidence = max_iterations_;

	for (int itr = 0; itr < max_iterations_ && itr < max_iterations_with_confidence; itr++) {
		int inliers_cnt = 0;
		Mat H(3, 3, CV_64F);
		Mat H_P(3, 3, CV_64F);

		int n = points_1_.size();

		// Standared RANSAC one iteration
		// 1. select 4 random indices from the set of matched corresponding points
		int i = 0;
		vector<int> indices;
		indices.resize(4);
		indices[0] = indices[1] = indices[2] = indices[3] = -1;
		do {
			int r = rand() % n;
			for (int j = 0; j < i; j++) {
				if (indices[j] == r)
					r = -1;
			}
			if (r != -1) {
				indices[i] = r;
				i++;
			}
		} while (i < 4);

		// 2. calculate the homography estimation matrix
		calcHomographyMatrix(normalized_points_1_, normalized_points_2_, indices, H_P);
		H = T_2_.inv() * H_P * T_1_;

		// 3. find inliers
		Mat A = Mat::zeros(3, n, CV_64F);
		for (int i = 0; i < n; i++) {
			A.at<double>(0, i) = points_1_[i].x;
			A.at<double>(1, i) = points_1_[i].y;
			A.at<double>(2, i) = 1;
		}
		A = H * A;
		for (int i = 0; i < n; i++) {
			Point2f p_dest(A.at<double>(0, i) / A.at<double>(2, i), A.at<double>(1, i) / A.at<double>(2, i));

			double d = calcDistance(p_dest, points_2_[i]);

			if (ransac_threshold_ > d) {
				inliers_cnt++;
			}
		}

		// 4. check best solution
		if (inliers_cnt > best_inliers) {
			best_inliers = inliers_cnt;
            //Normalize:
	        H = H * (1.0 / H.at<double>(2, 2));
			H.copyTo(bestH_);

			max_iterations_with_confidence = getIterationNumber(n, best_inliers, 4, ransac_confidence_);

		}
		cout << itr << " " << best_inliers << endl;

	}

}

// Find best homography estimation using RANSAC
void ransacBestHomographyEstimation(
	vector<Point2f> &points_1_, // input, matched corresponding points
	vector<Point2f> &points_2_,
	const int max_iterations_, // input, ransac iterations number
	const double ransac_threshold_, // input, ransac threshold
	const double ransac_confidence_,
	Mat &bestH_  // output, best homgraphy estimation
){
    // Normalize points
	vector<Point2f> normalized_points_1, normalized_points_2;
	Mat T_1, T_2;
	normalizePoints(
		points_1_,
		normalized_points_1,
		T_1
	);
	normalizePoints(
		points_2_,
		normalized_points_2,
		T_2
	);

    // Find homography estimation
	int inliers;
	ransacBestHomographyEstimation(
		points_1_,
		points_2_,
		normalized_points_1,
		normalized_points_2,
		T_1,
		T_2,
		max_iterations_,
		ransac_threshold_,
		ransac_confidence_,
		bestH_
	);
}

void readMeasuerementMatrix(string file_name, vector<Point2f> &pnt_1, vector<Point2f> &pnt_2) {
	std::ifstream file_stream(file_name);

	vector<double> vec;
	double e;

	while (file_stream >> e) {
		vec.push_back(e);
	}
	file_stream.close();


	int element_cnt = vec.size();
	int point_cnt = element_cnt / 4;

	for (int i = 0; i < point_cnt; i++) {
		pnt_1.push_back(Point2d(vec[i], vec[i + point_cnt]));
		pnt_2.push_back(Point2d(vec[i + 2 * point_cnt], vec[i + 3 * point_cnt]));
	}

}

void writeMatrixToFile(Mat &A, string file_path){
    std::ofstream output_stream; 
	output_stream.open(file_path);

    int n = A.rows;
    int m = A.cols;

    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            output_stream << A.at<double>(i, j) << " ";
        }
        output_stream << endl;
    }

    output_stream.close();
}

int main(int argc, char** argv )
{
    // Setting the random seed to get random results in each run.
	srand(time(NULL));

	if (argc < 6)
	{
		cout << " Usage: implmentation_number(1 or 2), ransac_threshold, ransac_max_iterations, ransac_confidence, correspondences_file, homography_output_file" << endl;
        cout << "implmentation_number=1: own implmentation" << endl;
        cout << "implmentation_number=2: opencv implmentation" << endl;
		return -1;
	}
    string implmentation = argv[1];
    float ransac_threshold = stof(argv[2]);
    int ransac_max_iterations = stoi(argv[3]);
    float ransac_confidence = stof(argv[4]);
    string correspondences_file_path = argv[5];
    string homography_output_file = argv[6];

    vector<Point2f> points_1, points_2;
    readMeasuerementMatrix(correspondences_file_path, points_1, points_2);

    if(implmentation == "1"){
        Mat H(3, 3, CV_64F);
        ransacBestHomographyEstimation(
            points_1,
            points_2,
            ransac_max_iterations,
            ransac_threshold,
            ransac_confidence,
            H
        );
        writeMatrixToFile(H, homography_output_file);
    }else if(implmentation == "2"){
        
        Mat H = findHomography(
            points_1, 
            points_2, 
            RANSAC, // Use ransac for robustifcation
            ransac_threshold,
            noArray(),
            ransac_max_iterations,
            ransac_confidence
        );
        cout<<H<<endl;
        writeMatrixToFile(H, homography_output_file);
    }else{
		cout << " Usage: implmentation_number(1 or 2) ransac_threshold ransac_max_iterations ransac_confidence correspondences_file homography_output_file" << endl;
        cout << "implmentation_number=1: own implmentation" << endl;
        cout << "implmentation_number=2: opencv implmentation" << endl;
		return -1;
	}
}


