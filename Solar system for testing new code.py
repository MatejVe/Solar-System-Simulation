import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

'''
Ideje kako poboljšati kod:
1. Kreirati novu klasu satelit koja je pod gravitacijskim utjecajem ostalih planeta ali ona ne utječe na njih
2. Ta klasa ima metodu boost koja ju odjednom ubrza
3. Pogledati i probati dolje zakomentiranu metodu za optimizaciju
4. Dodati tragove orbita
'''


class Body(object):
    def __init__(self, name, colour, mass, init_position, init_velocity):
        self.name = name
        self.colour = colour
        self.mass = mass
        self.position = init_position
        self.velocity = init_velocity
        self.current_acceleration = 0  # TODO: think if better initial acceleration can be found
        self.previous_acceleration = 0

    def update_position(self, timestep):
        """Updates the position of the body"""
        new_position = self.position + self.velocity * timestep + 1 / 6 * (4 * self.current_acceleration
                                                                           - self.previous_acceleration) * timestep ** 2
        self.position = new_position

    def update_velocity(self, new_acceleration, timestep):
        """New acceleration is the acceleration in the next timestep. Updates the body velocity"""
        new_velocity = self.velocity + 1 / 6 * (2 * new_acceleration + 5 * self.current_acceleration
                                                - self.previous_acceleration) * timestep
        self.velocity = new_velocity

    def calc_KE(self):
        """Returns the kinetic energy of the body"""
        KE = 1 / 2 * self.mass * np.linalg.norm(self.velocity) ** 2
        return KE


class SmallBody(Body):
    def __init__(self, name, colour, mass, init_position, init_velocity, bodyOfInterest, bodyOfInterestPosition):
        super().__init__(name, colour, mass, init_position, init_velocity)
        self.closestDistance = np.linalg.norm(bodyOfInterestPosition - init_position)
        self.timeToBodyOfInterest = 0
        self.bodyOfInterest = bodyOfInterest


def distance_from_body(body1, body2):
    distancevec = body1.position - body2.position
    distance = np.linalg.norm(distancevec)
    return distance


class Simulation(object):
    def __init__(self, timestep, num_iterations):
        self.timestep = timestep
        self.num_iterations = num_iterations
        self.patches = []
        self.timeElapsed = 0

    def read_input_data(self, file):
        """Opens the specific file and reads the input data. File has to be in a specific format"""
        df = open(file, 'r')
        data = df.readlines()
        data.pop(0)  # pop the first two lines of the file, they describe how the file is to be formated
        data.pop(0)
        bodies = []
        smallBodies = []
        for line in data:
            line = line.strip()
            line = line.split(',')
            line[3] = line[3].split(';')
            line[4] = line[4].split(';')
            name, color, mass, init_position, init_velocity = line[0].strip(), line[1].strip(), float(line[2]), \
                                                              np.array([float(line[3][0].strip()),
                                                                        float(line[3][1].strip())]), \
                                                              np.array([float(line[4][0].strip()),
                                                                        float(line[4][1].strip())])
            if line[-1].strip() == 'Body':
                bodies.append(Body(name, color, mass, init_position, init_velocity))
            elif line[-1].strip() == 'SmallBody':
                bodyOfInterest = line[-2].strip()
                for body in bodies:
                    if body.name == bodyOfInterest:
                        bodyOfInterestPosition = body.position
                smallBodies.append(SmallBody(name, color, mass, init_position, init_velocity, bodyOfInterest,
                                             bodyOfInterestPosition))
        self.body_list = bodies
        self.smallBodies = smallBodies

        for body in self.body_list:
            # create patches for each body of the system
            xpos = body.position[0]
            ypos = body.position[1]
            if body.name == 'Sun':
                self.patches.append(plt.Circle((xpos, ypos), radius=10000000000, color=body.colour, animated=True))
            else:
                for i in range(10):
                    self.patches.append(
                        plt.Circle((xpos, ypos), radius=(5000000000 / (10 - i)), color=body.colour, animated=True))

        for smallBody in self.smallBodies:
            xpos = smallBody.position[0]
            ypos = smallBody.position[1]
            for i in range(10):
                self.patches.append(
                    plt.Circle((xpos, ypos), radius=(5000000000 / (10 - i)), color=smallBody.colour, animated=True)
                )

        xmax = 0
        for body in self.body_list:
            # find the axes range
            if body.position[0] > xmax:
                xmax = body.position[0] * 1.5
            if body.position[1] > xmax:
                xmax = body.position[1] * 1.5

        self.xmax = xmax

    def run_simulation(self):
        # running the simulation for the inputed number of iterations
        for i in range(self.num_iterations):
            self.step_forward()

    def step_forward(self):
        # Move the bodies one timestep
        # New positions of all the bodies are calculated first
        self.timeElapsed += self.timestep
        for body in self.body_list:
            body.update_position(self.timestep)

        for smallBody in self.smallBodies:
            smallBody.update_position(self.timestep)

        for body in self.body_list:
            new_acceleration = self.calc_acceleration(body)
            body.update_velocity(new_acceleration, self.timestep)
            body.previous_acceleration = body.current_acceleration
            body.current_acceleration = new_acceleration

        for smallBody in self.smallBodies:
            new_acceleration = self.calc_acceleration(smallBody)
            smallBody.update_velocity(new_acceleration, self.timestep)
            smallBody.previous_acceleration = smallBody.current_acceleration
            smallBody.current_acceleration = new_acceleration
            for body in self.body_list:
                if smallBody.bodyOfInterest == body.name:
                    distance = distance_from_body(smallBody, body)
                    if distance < smallBody.closestDistance:
                        smallBody.closestDistance = distance
                        smallBody.timeToBodyOfInterest = self.timeElapsed

    def calc_acceleration(self, body):
        # find the acceleration on a single body. Returns a np array of acceleration
        forceOnBody = np.array([0.0, 0.0])
        for secondBody in self.body_list:
            if secondBody.name != body.name:
                displacementVec = secondBody.position - body.position
                distance = np.linalg.norm(displacementVec)
                displacementVec = displacementVec / distance

                magnitude = G * body.mass * secondBody.mass / (distance ** 2)
                force = magnitude * displacementVec

                forceOnBody += force

        acceleration = forceOnBody / body.mass
        return acceleration

    def update_display(self, i):
        # single timestep change in display
        self.step_forward()

        j = 0
        for body in self.body_list:
            if body.name == 'Sun':
                self.patches[j].center = (body.position[0], body.position[1])
            else:
                for i in range(1, 10):
                    self.patches[(j - 1) * 10 + i].center = self.patches[(j - 1) * 10 + i + 1].center
                self.patches[j * 10].center = (body.position[0], body.position[1])
            j += 1

        for smallBody in self.smallBodies:
            for i in range(1, 10):
                self.patches[(j - 1) * 10 + i].center = self.patches[(j - 1) * 10 + i + 1].center
            self.patches[j * 10].center = (smallBody.position[0], smallBody.position[1])
            j += 1
        return self.patches

    def animate(self):
        # animate the bodies for the duration of the simulation
        plt.style.use('dark_background')
        fig = plt.figure(1)
        ax = plt.axes()

        for patch in self.patches:
            ax.add_patch(patch)

        ax.axis('scaled')
        ax.set_xlim(-self.xmax, self.xmax)
        ax.set_ylim(-self.xmax, self.xmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.axis('off')

        anim = FuncAnimation(fig, self.update_display, frames=self.num_iterations, repeat=False, interval=50, blit=True)

        plt.show()

    def calc_PE(self):
        # Calculates the total potential energy. Returns a float of the energy
        PE = 0
        for body in self.body_list:
            for secondBody in self.body_list:
                if body.name != secondBody.name:
                    displacementVec = secondBody.position - body.position
                    distance = np.linalg.norm(displacementVec)

                    PE += -1 / 2 * G * body.mass * secondBody.mass / distance
        return PE

    def calc_tot_energy(self):
        # Calculates the total energy. Returns a float
        PE = self.calc_PE
        KE = 0
        for body in self.body_list:
            KE += body.calc_KE()
        return KE + PE

    def check_orbital_period(self, body):
        # Finds the orbital period of a given body using trigonometric functions. Returns a float
        orbital_period = 0
        while not 0 > np.arctan2(body.position[1], body.position[0]) > -0.01:
            self.step_forward()
            orbital_period += self.timestep
        orbital_period = orbital_period / 86400
        return orbital_period

    def launch_sattelite(self, name, colour, mass, launchBodyName, radius, initVelocity, launchOrientation,
                         interestBody):
        """This is a function that launches a satellite from a given body. Input parameters are name, colour, mass,
        name of the body from which the satellite is to be launched, distance from the center of the body where the
        satellite is launched, initial velocity, orientation of the launch - inner if launching from the side facing the
        Sun, outer otherwise, and name of the body the satellite is trying to reach"""
        for body in self.body_list:
            if interestBody == body.name:
                interestBodyName = body.name
                interestBodyPosition = body.position
            if launchBodyName == body.name:
                launchBody = body
        xBodyPos = launchBody.position[0]
        yBodyPos = launchBody.position[1]
        angle = np.arctan2(yBodyPos, xBodyPos)
        if launchOrientation == 'inner':
            xOffset = -1 * np.tan(angle) * radius
            yOffset = -1 * np.tan(angle) * radius
            launchPosition = np.array([xBodyPos + xOffset, yBodyPos + yOffset])
            self.smallBodies.append(SmallBody(name, colour, mass, launchPosition, initVelocity, interestBodyName,
                                              interestBodyPosition))
            for i in range(10):
                self.patches.append(
                    plt.Circle((launchPosition[0], launchPosition[1]), radius=(2000000000 / (10 - i)),
                               color=colour, animated=True))
        elif launchOrientation == 'outer':
            xOffset = np.tan(angle) * radius
            yOffset = np.tan(angle) * radius
            launchPosition = np.array([xBodyPos + xOffset, yBodyPos + yOffset])
            self.smallBodies.append(SmallBody(name, colour, mass, launchPosition, initVelocity, interestBodyName,
                                              interestBodyPosition))
            for i in range(10):
                self.patches.append(
                    plt.Circle((launchPosition[0], launchPosition[1]), radius=(2000000000 / (10 - i)),
                               color=colour, animated=True))


G = 6.67408e-11

Universe = Simulation(200000, 2000)
Universe.read_input_data('Parameters.txt')
Universe.animate()
# print('Orbital period of the Earth is: ' + str(Universe.check_orbital_period(Universe.body_list[3])) + ' days')
# print(str(Universe.calc_tot_energy()))
# Universe.run_simulation()
