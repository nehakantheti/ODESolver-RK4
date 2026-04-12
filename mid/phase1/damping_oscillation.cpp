/*
F = ma = -kx
mx" = -kx - cx'

where v = x'
and a = x"

*/

#include<iostream>
#include<vector>
#include<fstream>
#include<cmath>

using namespace std;

// State : this represents the svstem at anv single moment
struct State{
    double x; // represents displacement at a given instance
    double v; // represents velocity at a given instance
};

// overloading addition operator
State operator+(const State& a, const State& b){
    return {a.x + b.x, a.v + b.v};
}

// multiplying with a scalar
State operator*(const State& a, double val){
    return {a.x * val, a.v * val};
}

// Svstem dvnamics - returns the derivative for a given state, with constants k, m and c
State dynamics(const State& s, double t){
    double k = 1.0;
    double m = 1.0;
    double c = 0.1;

    // derivatives of disp and velocity (acc to formulae)
    State d;
    d.x = s.v;
    d.v = -(k/m)*s.x - (c/m)*s.v;
    return d;
}

// dt is the step size used in eulers method(forward eulers method)
State solveEuler(const State& current, double t, double dt){
    State derivative = dynamics(current, t);
    // refer to eulers formula - yk+1 = yk + y'k * h
    return current + (derivative * dt);
}

State solveRK4(const State& current, double t, double dt){
    State k1 = dynamics(current, t);

    State k2_state = current + (k1 * (dt / 2.0));
    State k2 = dynamics(k2_state, t + (dt / 2.0));

    State k3_state = current + (k2 * (dt / 2.0));
    State k3 = dynamics(k3_state, t + (dt / 2.0));

    State k4_state = current + (k3 * dt);
    State k4 = dynamics(k4_state, t + dt);

    // rk4 formula - y + (dt/6) * (k1 + 2k2 + 2k3 + k4)
    State slope_avg = (k1 + (k2 * 2.0) + (k3 * 2.0) + k4) * (1.0 / 6.0);
    return current + (slope_avg * dt);
}

int main(){
    double t_max = 20.0;
    double dt = 0.1; // step size

    std::ofstream file("results.csv");
    file << "t,euler_x,rk4_x\n";

    State state_euler = {1.0, 0.0};
    State state_rk4 = {1.0, 0.0};

    for(double t = 0; t <= t_max; t += dt){
        file << t << "," << state_euler.x << "," << state_rk4.x << "\n";

        state_euler = solveEuler(state_euler, t, dt);
        state_rk4 = solveRK4(state_rk4, t, dt);
    }

    file.close();
    cout << "Simulation complete. Data save to results.csv !\n";
    return 0;
}