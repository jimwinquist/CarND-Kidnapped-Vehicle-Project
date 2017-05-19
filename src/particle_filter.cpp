/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  if (!initialized()) {
    num_particles = 100;
    default_random_engine gen;

    // initialize normal distribution for x, y, and theta centered around the given gps measurement
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    particles.clear();
    weights.clear();
    //
    for (int i=0; i < num_particles; i++) {
      // Create particles from random normal distribution and add Gaussian noise
      // Set uniform weights
      Particle p;
      p.id = i;
      p.x = dist_x(gen);
      p.y = dist_y(gen);
      p.theta = dist_theta(gen);
      p.weight = 1.0/num_particles;

      particles.push_back(p);
      weights.push_back(p.weight);
    }

    is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;
  for (int i=0; i < num_particles; i++) {
    // Create temporary variables for doing computation
    double p_x, p_y, p_theta, v_dot;
    p_x = particles[i].x;
    p_y = particles[i].y;
    p_theta = particles[i].theta;
    v_dot = velocity * delta_t;

    // update each particle using the bicyclye model for straight driving and driving while turning
    double pred_x, pred_y, pred_theta;
    if (fabs(yaw_rate) < 0.001) {
      pred_x = p_x + v_dot * cos(p_theta);
      pred_y = p_y + v_dot * sin(p_theta);
      pred_theta = p_theta;
    }
    else {
      pred_x = p_x + (velocity/yaw_rate) * (sin(p_theta + (yaw_rate * delta_t)) - sin(p_theta));
      pred_y = p_y + (velocity/yaw_rate) * (cos(p_theta) - cos(p_theta + (yaw_rate * delta_t)));
      pred_theta = p_theta + (yaw_rate * delta_t);
    }

    // Create a normal distribution for the predicted x, y, and theta
    normal_distribution<double> dist_x(pred_x, std_pos[0]);
    normal_distribution<double> dist_y(pred_y, std_pos[1]);
    normal_distribution<double> dist_theta(pred_theta, std_pos[2]);

    // Add a small amount of Gaussian noise to the predicted position and heading
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for (int i = 0; i < observations.size(); i++) {
    double closest_distance = numeric_limits<double>::max();

    for (int j = 0; j < predicted.size(); j++) {
      double current_distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      if (current_distance < closest_distance) {
        // Set the observation id to the closest landmark index
        observations[i].id = j;
        closest_distance = current_distance;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

  // For Each Particle
  double sum_weights = 0.0;
  for (int i=0; i < num_particles; i++) {
    Particle p = particles[i];

    // Convert observations from vehicle to map coordinates
    vector<LandmarkObs> transformed_observations;
    for (int j=0; j < observations.size(); j++){
      LandmarkObs trans;
      trans.id = 0;
      trans.x = (observations[j].x * cos(p.theta)) - (observations[j].y * sin(p.theta)) + p.x;
      trans.y = (observations[j].x * sin(p.theta)) + (observations[j].y * cos(p.theta)) + p.y;
      transformed_observations.push_back(trans);
    }

    // Find map_landmarks in sensor_range
    vector<LandmarkObs> predicted;
    for (int k=0; k < map_landmarks.landmark_list.size(); k++) {
      if (dist(p.x, p.y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f) <= sensor_range) {
        LandmarkObs prediction;
        prediction.id = map_landmarks.landmark_list[k].id_i;
        prediction.x = map_landmarks.landmark_list[k].x_f;
        prediction.y = map_landmarks.landmark_list[k].y_f;
        predicted.push_back(prediction);
      }
    }

    // Associate landmarks with observations
    dataAssociation(predicted, transformed_observations);

    // Calculate particle weights
    double new_weight = 1.0;
    for (int m=0; m<transformed_observations.size(); m++) {
      // Create temporary variables for doing calculations
      int index = transformed_observations[m].id;
      double delta_x = predicted[index].x - transformed_observations[m].x;
      double delta_y = predicted[index].y - transformed_observations[m].y;

      double c1 = pow(delta_x, 2);
      double c2 = pow(delta_y, 2);
      double c3 = 2.0 * pow(std_landmark[0], 2);
      double c4 = 2.0 * pow(std_landmark[1], 2);
      double c5 = 2.0 * M_PI * std_landmark[0] * std_landmark[1];
      double c6 = (-c1/c3 - c2/c4);

      // Calculate updated weight value
      new_weight *= exp(c6) / c5;
    }
    // Update particle weights
    particles[i].weight = new_weight;
    weights[i] = new_weight;

    // Store the sum of the weights for normalization
    sum_weights += new_weight;
  }

  // Normalize weights
  if (sum_weights > 0) {
    for (int i = 0; i < particles.size(); i++) {
      particles[i].weight /= sum_weights;
      weights[i] /= sum_weights;
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;

  // Create discrete distribution from weights
  discrete_distribution<int> dist_particles(weights.begin(), weights.end());

  weights.clear();
  vector<Particle> resampled;
  // resample particles from the discrete weight distribution and update weights
  for (int i=0; i < num_particles; i++) {
    int index = dist_particles(gen);
    Particle p;
    p.id = particles[index].id;
    p.x = particles[index].x;
    p.y = particles[index].y;
    p.theta = particles[index].theta;
    p.weight = particles[index].weight;
    resampled.push_back(p);
    weights.push_back(p.weight);
  }

  // update particles to the resampled set
  particles = resampled;

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
