/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::default_random_engine;
using std::normal_distribution;
using std::discrete_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * TODO: Set the number of particles. Initialize all particles to
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1.
     * TODO: Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method
     *   (and others in this file).
     */
    num_particles = 50;  // TODO: Set the number of particles
    default_random_engine gen;
    is_initialized = false;
    
    
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    for(int i = 0 ; i < num_particles; i++)
    {
        //adding particle to vector of particles
        particles.push_back(Particle{i,dist_x(gen),dist_y(gen),dist_theta(gen),1.0});
        
        //adding weight to vector of weights
        weights.push_back(1.0);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
    /**
     * TODO: Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */
    default_random_engine gen;
    
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);
    
    for(int i = 0 ; i < num_particles; i++)
    {
        if (fabs(yaw_rate) < 0.00001) {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }
        else {
            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
            particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
            particles[i].theta += yaw_rate * delta_t;
        }
        
        // adding noise
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
    /**
     * TODO: Find the predicted measurement that is closest to each
     *   observed measurement and assign the observed measurement to this
     *   particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will
     *   probably find it useful to implement this method and use it as a helper
     *   during the updateWeights phase.
     */
    
    for (int i = 0; i < observations.size(); ++i)
    {
        int closest_landmark = 0;
        int min_dist = 999999;
        int curr_dist;
        // Iterate through all landmarks to check which is closest
        for (int j = 0; j < predicted.size(); ++j)
        {
            // Calculate Euclidean distance
            curr_dist = sqrt(pow(observations[i].x - predicted[j].x, 2)
                             + pow(observations[i].y - predicted[j].y, 2));
            // Compare to min_dist and update if closest
            if (curr_dist < min_dist)
            {
                min_dist = curr_dist;
                closest_landmark = predicted[j].id;
            }
        }
        observations[i].id = closest_landmark;
    }
    
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    /**
     * TODO: Update the weights of each particle using a mult-variate Gaussian
     *   distribution. You can read more about this distribution here:
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system.
     *   Your particles are located according to the MAP'S coordinate system.
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */
    
    double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    // calculate divisor required for calculating exponent
    double exp_divisor_x = 2 * std_landmark[0] * std_landmark[0];
    double exp_divisor_y = 2 * std_landmark[1] * std_landmark[1];
    
    
    for (int i = 0; i< num_particles ; i++)
    {
        vector<LandmarkObs> range_landmarks;
        
        for(int j=0; j < map_landmarks.landmark_list.size();j++)
        {
            if(dist(particles[i].x,particles[i].y,map_landmarks.landmark_list[j].x_f,map_landmarks.landmark_list[j].y_f) <= sensor_range)
            {
                LandmarkObs obs;
                obs.id = map_landmarks.landmark_list[j].id_i;
                obs.x = map_landmarks.landmark_list[j].x_f;
                obs.y = map_landmarks.landmark_list[j].y_f;
                
                range_landmarks.push_back(obs);
            }
        }
        
        vector<LandmarkObs> transformed_obs;
        for (unsigned int j = 0; j < observations.size(); j++) {
            double transformed_x = cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y + particles[i].x;
            double transformed_y = sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y + particles[i].y;
            transformed_obs.push_back(LandmarkObs{ observations[j].id, transformed_x, transformed_y });
        }
        
        dataAssociation(range_landmarks, transformed_obs);
        
        double weight = 1.0;
        for(unsigned int j = 0; j < transformed_obs.size(); j++) {
            for(unsigned int k = 0; k < range_landmarks.size(); k++) {
                if(range_landmarks[k].id == transformed_obs[j].id) {
                    double dx = range_landmarks[k].x - transformed_obs[j].x;
                    double dy = range_landmarks[k].y - transformed_obs[j].y;
                    weight *= gauss_norm * exp( -(((dx*dx) / exp_divisor_x) + ((dy*dy) / exp_divisor_y)) );
                    break;
                }
            }
        }
        particles[i].weight = weight;
        weights[i] = weight;
        
    }
    
}

void ParticleFilter::resample() {
    /**
     * TODO: Resample particles with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */
    default_random_engine gen;
    
    vector<Particle> new_particles;
    
    // generate distribution proportional to weight
    discrete_distribution<int> dist(weights.begin(), weights.end());
    
    // resample particles using above distribution
    for(int i = 0; i < num_particles; i++) {
        new_particles.push_back(particles[dist(gen)]);
    }
    
    particles = new_particles;
    
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
    vector<double> v;
    
    if (coord == "X") {
        v = best.sense_x;
    } else {
        v = best.sense_y;
    }
    
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
