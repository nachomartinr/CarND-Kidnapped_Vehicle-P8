/*
 * particle_filter.cpp
 *
 *  Created on: Jun 25, 2017
 *      Author: Ignacio Martin
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>



#include "particle_filter.h"

using namespace std;

std::random_device rnd_dev;
/* PRNG */
std::mt19937 rnd_gen(rnd_dev());

inline double bivariate_gaussian(double x, double y, double mu_x, double mu_y, double std_x, double std_y) {

    /* evaluate the bivariate normal distribution at (x,y), assuming that their covariance is 0. */

    return (1/(2*M_PI*std_x*std_y)) * exp(-(pow(x-mu_x, 2)/(2*pow(std_x,2)) +
                                            pow(y-mu_y, 2)/(2*pow(std_y,2))));
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {

    // Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;

    particles.clear();
    weights.clear();
    particles.reserve(num_particles);
    weights.reserve(num_particles);
  
    // normal distributions to model the uncertainty of the initial position
    std::normal_distribution<double> x_pdf(0, std[0]);
    std::normal_distribution<double> y_pdf(0, std[1]);
    std::normal_distribution<double> theta_pdf(0, std[2]);

    for (unsigned int i = 0; i < num_particles; ++i) {
        Particle p_temp;
        p_temp.id = i;
        p_temp.x = x + x_pdf(rnd_gen);
        p_temp.y = y + y_pdf(rnd_gen);
        p_temp.theta = theta + theta_pdf(rnd_gen);;
        p_temp.weight = 1.0;

        particles.push_back(std::move(p_temp));
        weights.push_back(p_temp.weight);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

    // Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
  
    // normal distributions of the process noise of each state component
    std::normal_distribution<double> x_pdf(0, std_pos[0]);
    std::normal_distribution<double> y_pdf(0, std_pos[1]);
    std::normal_distribution<double> theta_pdf(0, std_pos[2]);

    for (unsigned int i = 0; i < num_particles; ++i) {
        double x_new;
        double y_new;
        double theta_new;
        const double theta_0 = particles[i].theta;

        if (fabs(yaw_rate) > 0.00001) {

            x_new = particles[i].x + (velocity / yaw_rate) * (sin(theta_0 + (yaw_rate * delta_t)) - sin(theta_0));
            y_new = particles[i].y + (velocity / yaw_rate) * (cos(theta_0) - cos(theta_0 + (yaw_rate * delta_t)));
            theta_new = theta_0 + (yaw_rate * delta_t);
        }
        else {
            // yaw_rate ~ 0;
            x_new = particles[i].x + ((velocity * delta_t) * cos(theta_0));
            y_new = particles[i].y + ((velocity * delta_t) * sin(theta_0));
            theta_new = theta_0;
        }

        particles[i].x = x_new + x_pdf(rnd_gen);
        particles[i].y = y_new + y_pdf(rnd_gen);
        particles[i].theta = theta_new + theta_pdf(rnd_gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

    // Find the predicted measurement that is closest to each observed measurement and assign the 
    // observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

  for (unsigned int i = 0; i < observations.size(); ++i) {

        double min_dist = std::numeric_limits<double>::max();
        int closest_id = -1;
    
        for(unsigned int j = 0; j < predicted.size(); ++j) {

            const double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

            if (distance < min_dist) {
                min_dist = distance;
                closest_id = predicted[j].id;
            }
        }

        observations[i].id = closest_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
    // Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    for (unsigned int i = 0; i < num_particles; i++) {
        const double p_x = particles[i].x;
        const double p_y = particles[i].y;
        const double p_theta = particles[i].theta;

        // Transform observations from vehicle space to map space
        std::vector<LandmarkObs> trans_observations;

        for (unsigned int j = 0; j < observations.size(); ++j) {
            LandmarkObs trans_observation;

            // | x |   | cos(th) -sin(th) tx | | x'|
            // | y | = | sin(th)  cos(th) ty | | y'|
            // | 1 |   |   0        0      1 | | 1 |

            trans_observation.x = p_x + (observations[j].x * cos(p_theta)) - (observations[j].y * sin(p_theta));
            trans_observation.y = p_y + (observations[j].x * sin(p_theta)) + (observations[j].y * cos(p_theta)); 

            trans_observation.id = -1;


            trans_observations.push_back(trans_observation);
        }

        // Predicted observations for the current particle
        std::vector<LandmarkObs> pred_observations;

        for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {

            const double distance = dist(p_x,
                                         p_y,
                                         map_landmarks.landmark_list[j].x_f,
                                         map_landmarks.landmark_list[j].y_f);
      
            // filter out all map landmarks whose distance to the particle is greater than the sensor_range
            if (distance <= sensor_range) {                
                LandmarkObs pred_observation;
                pred_observation.id = map_landmarks.landmark_list[j].id_i;
                pred_observation.x = map_landmarks.landmark_list[j].x_f;
                pred_observation.y = map_landmarks.landmark_list[j].y_f;
                pred_observations.push_back(pred_observation);
            }
        }

        // Associate observed measurements to known map landmarks
        dataAssociation(pred_observations, trans_observations);


        vector<int> associations_vec;
        vector<double> sense_x_vec;
        vector<double> sense_y_vec;

        /* Calculate weights based on the likelihood of the observations */
        double particle_weight = 1.0;

        // Get coordinates of each transformed observation
        for (unsigned int j = 0; j < trans_observations.size(); ++j){
            const int id_obs = trans_observations[j].id;
            const double x_obs = trans_observations[j].x;
            const double y_obs = trans_observations[j].y;
            double x_pred;
            double y_pred;
            bool id_found = false;

            /* Get the predicted observation associated with the current observation */
            for (unsigned int k = 0; k < pred_observations.size(); ++k) {
                if (pred_observations[k].id == id_obs){
                    x_pred = pred_observations[k].x;
                    y_pred = pred_observations[k].y;
                    id_found = true;
                }
            }
            
            if (id_found == true) {
                const double prob = bivariate_gaussian(x_obs, y_obs, x_pred, y_pred, std_landmark[0], std_landmark[1]);

                particle_weight *= prob;

                associations_vec.push_back(id_obs);
                sense_x_vec.push_back(x_obs);
                sense_y_vec.push_back(y_obs); 
            }

        }

        particles[i].weight = particle_weight;
        weights[i] = particle_weight;

        SetAssociations(particles[i], associations_vec, sense_x_vec, sense_y_vec);

    }
}

void ParticleFilter::resample() {
    // Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::discrete_distribution<int> ddistr(weights.begin(), weights.end());

    std::vector<Particle> resampled_particles;

    for (unsigned int i=0; i<num_particles; i++){

        const int sample_idx = ddistr(rnd_gen);
        resampled_particles.push_back(particles[sample_idx]);
    }

    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y) {
  
    // particle: the particle to assign each listed association, and association's
    // (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    // Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best) {

    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best) {

    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best) {

    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
