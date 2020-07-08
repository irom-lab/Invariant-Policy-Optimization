"""
Things to do:
- 
"""


from gym_minigrid.minigrid import *
from gym_generalization.register import register

class ColorableKey(Key):
    def __init__(self, color='red', is_colorable=True):
        super(ColorableKey, self).__init__(color)
        self.is_colorable = is_colorable

    def toggle(self, env, pos):
        # Change the color of the key
        if self.is_colorable and (self.color == 'red'):
            self.color = 'yellow'
        if self.is_colorable and (self.color == 'yellow'):
            self.color = 'red' 

        return True

class ColorblindDoor(Door):
    def __init__(self, color='grey', is_open=False, is_locked=False):
        super(ColorblindDoor, self).__init__(color, is_open, is_locked)

    def toggle(self, env, pos):
        # If the player has any key, it can open the door
        if self.is_locked:
            if isinstance(env.carrying, Key):
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True


class ColorableKeysEnv(MiniGridEnv):
    """
    Environment for testing generalization to novel environments that are drawn from a different distribution.
    During training: the robot must find a key to open a door. The keys are always colored red. But the robot
    can change the color of the keys.
    During test: the robot must again find a key to open a door. But the keys are colored differently. The robot
    cannot change the color of the keys.
    """

    def __init__(self, color='yellow', is_colorable=True, size=5):
        self.key_color = color
        self.is_colorable = is_colorable
        super().__init__(
            grid_size=size,
            max_steps=10*size*size
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a (colorblind) door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(ColorblindDoor('grey', is_locked=True), splitIdx, doorIdx)
        # self.put_obj(Door(self.key_color, is_locked=True), splitIdx, doorIdx)

        # Place a key on the left side
        self.place_obj(
            obj=ColorableKey(self.key_color, self.is_colorable),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.mission = "use the key to open the door and then get to the goal"

class ColorableKeysTrain(ColorableKeysEnv):
    def __init__(self):
        super().__init__(color='red', is_colorable=True)

class ColorableKeysTest(ColorableKeysEnv):
    def __init__(self):
        super().__init__(color='yellow', is_colorable=False)

register(
    id='MiniGrid-ColorableKeysTrain-v0',
    entry_point='gym_generalization.envs:ColorableKeysTrain'
)

register(
    id='MiniGrid-ColorableKeysTest-v0',
    entry_point='gym_generalization.envs:ColorableKeysTest'
)
