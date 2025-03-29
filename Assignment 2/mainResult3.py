import numpy as np
import matplotlib.pyplot as plt
import random

class StateModel:

    def __init__(self, rows, cols):
        self.__rows = rows
        self.__cols = cols
        self.__head = 4
        self.__num_states = rows*cols*4
        self.__num_readings = rows*cols+1

    def state_to_pose(self, s: int) -> (int, int, int):
        x = s // (self.__cols * self.__head)
        y = (s - x * self.__cols * self.__head) // self.__head
        h = s % self.__head

        return x, y, h;

    def pose_to_state(self, x: int, y: int, h: int) -> int:
        return x * self.__cols * self.__head + y * self.__head + h

    def state_to_position(self, s: int) -> (int, int):
        x = s // (self.__cols * self.__head)
        y = (s - x * self.__cols * self.__head) // self.__head

        return x, y

    def reading_to_position(self, r: int) -> (int, int):
        x = r // self.__cols
        y = r % self.__cols

        return x, y

    def position_to_reading(self, x: int, y: int) -> int:
        return x * self.__cols + y

    def state_to_reading(self, s: int) -> int:
        return s // self.__head

    # Note that a reading contains less information than a state, i.e., this will always give the state
    # corresponding to heading=0 (SOUTH) in the cell that corresponds to "reading r"
    def reading_to_ref_state(self, r: int) -> int:
        return r * self.__head

    def get_grid_dimensions(self) -> (int, int, int):
        return self.__rows, self.__cols, self.__head

    def get_num_of_states(self) -> int:
        return self.__num_states

    def get_num_of_readings(self) -> int:
        return self.__num_readings
    
    
class TransitionModel:
    def __init__(self, stateModel):
        self.__sm = stateModel
        self.__rows, self.__cols, self.__head = self.__sm.get_grid_dimensions()

        self.__dim = self.__rows * self.__cols * self.__head

        self.__matrix = np.zeros(shape=(self.__dim, self.__dim), dtype=float)
        for i in range(self.__dim):
            x, y, h = self.__sm.state_to_pose(i)
            for j in range(self.__dim):
                nx, ny, nh = self.__sm.state_to_pose(j)

                # If the new position is one step away in a "legal" direction
                if abs(x - nx) + abs(y - ny) == 1 and \
                        (nh == 2 and nx == x - 1 or nh == 1 and ny == y + 1 or \
                         nh == 0 and nx == x + 1 or nh == 3 and ny == y - 1):
                    
                    # entry where new and old heading are the same
                    if nh == h:
                        self.__matrix[i, j] = 0.7

                    else: # entry where new and old heading are different, i.e., distributing probabilities for the "rest"
                        if x != 0 and x != self.__rows - 1 and y != 0 and y != self.__cols - 1:
                            self.__matrix[i, j] = 0.1

                        # Facing a wall, not in a corner    
                        elif h == 2 and x == 0 and y != 0 and y != self.__cols - 1 or \
                                h == 1 and x != 0 and x != self.__rows - 1 and y == self.__cols - 1 or \
                                h == 0 and x == self.__rows - 1 and y != 0 and y != self.__cols - 1 or \
                                h == 3 and x != 0 and x != self.__rows - 1 and y == 0:

                            self.__matrix[i, j] = 1.0 / 3.0
                            
                        # Going along a wall 
                        elif h != 2 and x == 0 and y != 0 and y != self.__cols - 1 or \
                                h != 1 and x != 0 and x != self.__rows - 1 and y == self.__cols - 1 or \
                                h != 0 and x == self.__rows - 1 and y != 0 and y != self.__cols - 1 or \
                                h != 3 and x != 0 and x != self.__rows - 1 and y == 0:

                            self.__matrix[i, j] = 0.15

                        # In a corner, facing wall    
                        elif (h == 2 or h == 3) and (nh == 1 or nh == 0) and x == 0 and y == 0 or \
                                (h == 2 or h == 1) and (nh == 0 or nh == 3) and x == 0 and y == self.__cols - 1 or \
                                (h == 1 or h == 0) and (nh == 2 or nh == 3) and x == self.__rows - 1 and y == self.__cols - 1 or \
                                (h == 0 or h == 3) and (nh == 2 or nh == 1) and x == self.__rows - 1 and y == 0:

                            self.__matrix[i, j] = 0.5

                        # In a corner, not facing wall    
                        elif (h == 0 and nh == 1 or h == 1 and nh == 0) and x == 0 and y == 0 or \
                                (h == 0 and nh == 3 or h == 3 and nh == 0) and x == 0 and y == self.__cols - 1 or \
                                (h == 2 and nh == 1 or h == 1 and nh == 2) and x == self.__rows - 1 and y == 0 or \
                                (h == 2 and nh == 3 or h == 3 and nh == 2) and x == self.__rows - 1 and y == self.__cols - 1:

                            self.__matrix[i, j] = 0.3
                            
        # if we only have one row or colum in the grid, but more than 1 cells
        if (self.__rows == 1 or self.__cols == 1) and self.__rows * self.__cols != 1:  
            for i in range(self.__dim):
                sum = np.sum(self.__matrix[i, :])
                self.__matrix[i, :] = self.__matrix[i, :] / sum

    # retrieve the number of states represented in the matrix
    def get_num_of_states(self) -> int:
        return self.__dim

    # get the probability to go from state i to j
    def get_T_ij(self, i: int, j: int) -> float:
        return self.__matrix[i, j]

    # get the entire matrix (dimensions: nr_of_states x nr_of_states, type float)
    def get_T(self) -> np.array(2):
        return self.__matrix.copy()

    # get the transposed transition matrix (dimensions: nr_of_states x nr_of_states, type float)
    def get_T_transp(self) -> np.array(2):
        transp = np.transpose(self.__matrix)
        return transp

    # plot matrix as a heat map
    def plot_T(self):
        plt.matshow(self.__matrix)
        plt.colorbar()
        plt.show()
        
class ObservationModel:
    def __init__(self, stateModel):

        self.__stateModel = stateModel
        self.__rows, self.__cols, self.__head = stateModel.get_grid_dimensions()

        self.__dim = self.__rows * self.__cols * self.__head
        self.__num_readings = self.__rows * self.__cols + 1

        self.__vectors = np.ones(shape=(self.__num_readings, self.__dim))

        for o in range(self.__num_readings - 1):
            sx, sy = self.__stateModel.reading_to_position(o)

            for i in range(self.__dim):
                x, y = self.__stateModel.state_to_position(i)
                self.__vectors[o, i] = 0.0

                if x == sx and y == sy:
                    # "correct" reading 
                    self.__vectors[o, i] = 0.1
                elif (x == sx + 1 or x == sx - 1) and y == sy:
                    # first ring, below or above
                    self.__vectors[o, i] = 0.05
                elif (x == sx + 1 or x == sx - 1) and (y == sy + 1 or y == sy - 1):
                    # first ring, "corners"
                    self.__vectors[o, i] = 0.05
                elif x == sx and (y == sy + 1 or y == sy - 1):
                    # first ring, left or right
                    self.__vectors[o, i] = 0.05
                elif (x == sx + 2 or x == sx - 2) and (y == sy or y == sy + 1 or y == sy - 1):
                    # second ring, above / below / left / right
                    self.__vectors[o, i] = 0.025
                elif (x == sx + 2 or x == sx - 2) and (y == sy + 2 or y == sy - 2):
                    # second ring, "corners"
                    self.__vectors[o, i] = 0.025
                elif (x == sx or x == sx + 1 or x == sx - 1) and (y == sy + 2 or y == sy - 2):
                    # second ring, "horse" metric
                    self.__vectors[o, i] = 0.025

                self.__vectors[self.__num_readings - 1, i] -= self.__vectors[o, i];  # sensor reading "nothing"

    # get the number of possible sensor readings (rows * columns + 1)
    def get_nr_of_readings(self) -> int:
        return self.__num_readings

    # get the probability for the sensor to have produced reading "reading" when in state "state"
    def get_o_reading_state(self, reading: int, i: int) -> float:
        if reading == None : reading = self.__num_readings-1
        return self.__vectors[reading, i]

    # get the diagonale matrix O_reading with probabilities of the states i, i=0...nrOfStates-1 
    # to have produced reading "reading", returns a 2d-float array
    # use None for "no reading"
    def get_o_reading(self, reading: int) -> np.array(2):
        if (reading == None): reading = self.__num_readings - 1
        return np.diag( self.__vectors[reading, :])

    def sum_diags(self):
        dummy = np.diag( self.__vectors[0,:])
        for i in range(1, self.__num_readings):
            dummy += np.diag(self.__vectors[i,:])
        return dummy

    # plot the vectors as heat map(s)
    def plot_o_diags(self):
        plt.matshow(self.__vectors)
        plt.colorbar()
        plt.show()
        
class ObservationModelUF:
    def __init__(self, stateModel):

        self.__stateModel = stateModel
        self.__rows, self.__cols, self.__head = stateModel.get_grid_dimensions()

        self.__dim = self.__rows * self.__cols * self.__head
        self.__num_readings = self.__rows * self.__cols + 1

        self.__vectors = np.ones(shape=(self.__num_readings, self.__dim))

        for o in range(self.__num_readings - 1):
            x, y = self.__stateModel.reading_to_position(o)

            for i in range(self.__dim):
                sx, sy = self.__stateModel.state_to_position(i)
                self.__vectors[o, i] = 0.0

                if x == sx and y == sy:
                    # "correct" reading 
                    self.__vectors[o, i] = 0.1

                elif (x == sx + 1 or x == sx - 1) and y == sy:
                    # first ring, below or above
                    if ( sx == 0 and (sy == 0 or sy == self.__cols-1)) \
                        or (sx == self.__rows-1 and (sy == 0 or sy == self.__cols-1)):
                        self.__vectors[o, i] += 0.4/3
                    elif ( sx == 0 or sx == self.__rows - 1 or sy == 0 or sy == self.__cols-1):
                        self.__vectors[o, i] += 0.4/5
                    else:
                        self.__vectors[o, i] = 0.05
                elif (x == sx + 1 or x == sx - 1) and (y == sy + 1 or y == sy - 1):
                    # first ring, "corners"
                    if ( sx == 0 and (sy == 0 or sy == self.__cols-1)) \
                        or (sx == self.__rows-1 and (sy == 0 or sy == self.__cols-1)):
                        self.__vectors[o, i] += 0.4/3
                    elif ( sx == 0 or sx == self.__rows - 1 or sy == 0 or sy == self.__cols-1):
                        self.__vectors[o, i] += 0.4/5
                    else:
                        self.__vectors[o, i] = 0.05
                elif x == sx and (y == sy + 1 or y == sy - 1):
                    # first ring, left or right
                    if ( sx == 0 and (sy == 0 or sy == self.__cols-1)) \
                        or (sx == self.__rows-1 and (sy == 0 or sy == self.__cols-1)):
                        self.__vectors[o, i] += 0.4/3
                    elif ( sx == 0 or sx == self.__rows - 1 or sy == 0 or sy == self.__cols-1):
                        self.__vectors[o, i] += 0.4/5
                    else:
                        self.__vectors[o, i] = 0.05
                elif (x == sx + 2 or x == sx - 2) and (y == sy or y == sy + 1 or y == sy - 1):
                    # second ring, above / below / left / right
                    if (sx == 0 and (sy == 0 or sy == self.__cols-1)) \
                        or (sx == self.__rows-1 and (sy == 0 or sy == self.__cols-1)):
                        self.__vectors[o, i] += 0.4/5
                    elif (sx == 0 and (sy == 1 or sy == self.__cols-2)) \
                        or (sx == self.__rows-1 and (sy == 1 or sy == self.__cols-2)) \
                        or (sy == 0 and (sx == 1 or sx == self.__rows-2)) \
                        or (sy == self.__cols-1 and (sx == 1 or sx == self.__rows-2)):
                        self.__vectors[o, i] += 0.4/6
                    elif (sx == 1 and (sy == 1 or sy == self.__cols-2)) \
                        or (sx == self.__rows-2 and (sy == 1 or sy == self.__cols-2)):
                        self.__vectors[o, i] += 0.4/7
                    elif (sx == 0 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sx == self.__rows-1 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sy == 0 and (sx >= 2 and sx <= self.__rows-3)) \
                        or (sy == self.__cols-1 and (sx >= 2 and sx <= self.__rows-3)):
                        self.__vectors[o, i] += 0.4/9
                    elif (sx == 1 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sx == self.__rows-2 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sy == 1 and (sx >= 2 and sx <= self.__rows-3)) \
                        or (sy == self.__cols-2 and (sx >= 2 and sx <= self.__rows-3)):
                        self.__vectors[o, i] += 0.4/11
                    else:
                        self.__vectors[o, i] = 0.025
                elif (x == sx + 2 or x == sx - 2) and (y == sy + 2 or y == sy - 2):
                    # second ring, "corners"
                    if (sx == 0 and (sy == 0 or sy == self.__cols-1)) \
                        or (sx == self.__rows-1 and (sy == 0 or sy == self.__cols-1)):
                        self.__vectors[o, i] += 0.4/5
                    elif (sx == 0 and (sy == 1 or sy == self.__cols-2)) \
                        or (sx == self.__rows-1 and (sy == 1 or sy == self.__cols-2)) \
                        or (sy == 0 and (sx == 1 or sx == self.__rows-2)) \
                        or (sy == self.__cols-1 and (sx == 1 or sx == self.__rows-2)):
                        self.__vectors[o, i] += 0.4/6
                    elif (sx == 1 and (sy == 1 or sy == self.__cols-2)) \
                        or (sx == self.__rows-2 and (sy == 1 or sy == self.__cols-2)):
                        self.__vectors[o, i] += 0.4/7
                    elif (sx == 0 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sx == self.__rows-1 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sy == 0 and (sx >= 2 and sx <= self.__rows-3)) \
                        or (sy == self.__cols-1 and (sx >= 2 and sx <= self.__rows-3)):
                        self.__vectors[o, i] += 0.4/9
                    elif (sx == 1 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sx == self.__rows-2 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sy == 1 and (sx >= 2 and sx <= self.__rows-3)) \
                        or (sy == self.__cols-2 and (sx >= 2 and sx <= self.__rows-3)):
                        self.__vectors[o, i] += 0.4/11
                    else:
                        self.__vectors[o, i] = 0.025
                        
                elif (x == sx or x == sx + 1 or x == sx - 1) and (y == sy + 2 or y == sy - 2):
                    # second ring, "horse" metric
                    if (sx == 0 and (sy == 0 or sy == self.__cols-1)) \
                        or (sx == self.__rows-1 and (sy == 0 or sy == self.__cols-1)):
                        self.__vectors[o, i] += 0.4/5
                    elif (sx == 0 and (sy == 1 or sy == self.__cols-2)) \
                        or (sx == self.__rows-1 and (sy == 1 or sy == self.__cols-2)) \
                        or (sy == 0 and (sx == 1 or sx == self.__rows-2)) \
                        or (sy == self.__cols-1 and (sx == 1 or sx == self.__rows-2)):
                        self.__vectors[o, i] += 0.4/6
                    elif (sx == 1 and (sy == 1 or sy == self.__cols-2)) \
                        or (sx == self.__rows-2 and (sy == 1 or sy == self.__cols-2)): # sx == self.___rows-2
                        self.__vectors[o, i] += 0.4/7
                    elif (sx == 0 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sx == self.__rows-1 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sy == 0 and (sx >= 2 and sx <= self.__rows-3)) \
                        or (sy == self.__cols-1 and (sx >= 2 and sx <= self.__rows-3)):
                        self.__vectors[o, i] += 0.4/9
                    elif (sx == 1 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sx == self.__rows-2 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sy == 1 and (sx >= 2 and sx <= self.__rows-3)) \
                        or (sy == self.__cols-2 and (sx >= 2 and sx <= self.__rows-3)):
                        self.__vectors[o, i] += 0.4/11
                    else:
                        self.__vectors[o, i] = 0.025

                self.__vectors[self.__num_readings - 1, i] = 0.1;  # sensor reading "nothing"

    # get the number of possible sensor readings (rows * columns + 1)
    def get_nr_of_readings(self) -> int:
        return self.__num_readings

    # get the probability for the sensor to have produced reading "reading" when in state "state"
    def get_o_reading_state(self, reading: int, i: int) -> float:
        if reading == None : reading = self.__num_readings-1
        return self.__vectors[reading, i]

    # get the diagonale matrix O_reading with probabilities of the states i, i=0...nrOfStates-1 
    # to have produced reading "reading", returns a 2d-float array
    # use None for "no reading"
    def get_o_reading(self, reading: int) -> np.array(2):
        if (reading == None): reading = self.__num_readings - 1
        return np.diag( self.__vectors[reading, :])

    def sum_diags(self):
        dummy = np.diag( self.__vectors[0,:])
        for i in range(1, self.__num_readings):
            dummy += np.diag(self.__vectors[i,:])
        return dummy
        
    # plot the vectors as heat map(s)
    def plot_o_diags(self):
        plt.matshow(self.__vectors)
        plt.colorbar()
        plt.show()

class RobotSim:
    def __init__(self, state, sm):
        self.__currentState = state
        #print("starting in state {}".format(state))
        self.__sm = sm
        
    def move_once( self, tm) -> int:
        newState = -1
        rand = random.random()
        
        probSum = 0.0

        for i in range(self.__sm.get_num_of_states()) :
            probSum += tm.get_T_ij( self.__currentState, i)

            if( probSum > rand) :
                newState = i
                rand = 1.0
                break

        if( newState == -1) :
            print( " no new state found ")
        
        self.__currentState = newState;
        
        return self.__currentState
    
    def sense_in_current_state(self, om) -> int:
        sensorReading = None
        
        rand = random.random()
        
        probSum = 0.0

        for i in range(self.__sm.get_num_of_readings()-1) :
            probSum += om.get_o_reading_state( i, self.__currentState)

            if( probSum > rand) :
                sensorReading = i
                rand = 1.0
                break

        return sensorReading
    
class HMMFilter:
    def __init__(self, probs, tm, om, sm):
        self.__tm = tm
        self.__om = om
        self.__sm = sm
        self.__f = probs
        
    # sensorR is the sensor reading (index!), self._f is the probability distribution resulting from the filtering    
    def filter(self, sensorR : int) -> np.array :
        T = self.__tm.get_T_transp()
        Od = self.__om.get_o_reading(sensorR)
        self.__f = Od @ T @ self.__f
        self.__f /= np.sum(self.__f)
        return self.__f
    

def main():
    height = 20
    width = 16
    sm = StateModel(width, height)
    tm = TransitionModel(sm)
    omUF = ObservationModelUF(sm)
    omNUF = ObservationModel(sm)
    
    trueState = random.randint(0, sm.get_num_of_states() - 1)
    rs = RobotSim(trueState, sm)
    sense = None
    sense2 = None
    
    
    probs = np.ones(sm.get_num_of_states()) / (sm.get_num_of_states())
    probs2 = np.ones(sm.get_num_of_states()) / (sm.get_num_of_states())
    estimate = sm.state_to_position(np.argmax(probs))
    estimate2 = estimate
    
    
    
    filterUF = HMMFilter(probs, tm, omUF, sm)
    filterNUF = HMMFilter(probs2, tm, omNUF, sm)
    
    error1 = 0.0
    error2 = 0.0
    n = 500
    distances1 = np.zeros(n)
    distances2 = np.zeros(n)
    for i in range(n):
        trueState = rs.move_once(tm)
        sense = rs.sense_in_current_state(omNUF)
        sense2 = rs.sense_in_current_state(omUF)
        probs = filterNUF.filter(sense)
        probs2 = filterUF.filter(sense2)
       
        
        fPositions = probs.copy()
        fPositions2 = probs2.copy()
        for state in range(0, sm.get_num_of_states(), 4) :
            fPositions[state:state+4] = sum(fPositions[state:state+4])   
            fPositions2[state:state+4] = sum(fPositions2[state:state+4]) 
           
            
        estimate = sm.state_to_position(np.argmax(fPositions))
        eX, eY = estimate
        
        estimate2 = sm.state_to_position(np.argmax(fPositions2))
        eX2, eY2 = estimate2
        
        tsX, tsY = sm.state_to_position(trueState)
       
        error1 += abs(tsX-eX)+abs(tsY-eY)
        error2 += abs(tsX-eX2)+abs(tsY-eY2)
        distances1[i] = error1/(i+1)
        distances2[i] = error2/(i+1)
        
    plt.plot(range(1,n+1,1), distances1, label='Forward Filtering with NUF')
    plt.plot(range(1,n+1,1),distances2, label='Forward Filtering with UF')
    plt.legend(loc="upper right")
    plt.show()
                       

if __name__ == "__main__":
    main()    