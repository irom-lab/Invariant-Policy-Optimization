from gym_minigrid.minigrid import *
from gym_generalization.register import register


class ColoredKeysEnv(MiniGridEnv):
    """
    Simple environment for testing generalization to novel environments that are drawn from a different distribution.
    """

    def __init__(self, color='yellow', size=5):
        self.key_color = color
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

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door(self.key_color, is_locked=True), splitIdx, doorIdx)

        # Place a key on the left side
        self.place_obj(
            obj=Key(self.key_color),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.mission = "use the key to open the door and then get to the goal"

class ColoredKeysRed(ColoredKeysEnv):
    def __init__(self):
        super().__init__(color='red')

class ColoredKeysYellow(ColoredKeysEnv):
    def __init__(self):
        super().__init__(color='yellow')

class ColoredKeysGreen(ColoredKeysEnv):
    def __init__(self):
        super().__init__(color='green')     

class ColoredKeysBlue(ColoredKeysEnv):
    def __init__(self):
        super().__init__(color='blue')         

class ColoredKeysPurple(ColoredKeysEnv):
    def __init__(self):
        super().__init__(color='purple')                

class ColoredKeysGrey(ColoredKeysEnv):
    def __init__(self):
        super().__init__(color='grey')    

register(
    id='MiniGrid-ColoredKeysRed-v0',
    entry_point='gym_generalization.envs:ColoredKeysRed'
)

register(
    id='MiniGrid-ColoredKeysYellow-v0',
    entry_point='gym_generalization.envs:ColoredKeysYellow'
)

register(
    id='MiniGrid-ColoredKeysGreen-v0',
    entry_point='gym_generalization.envs:ColoredKeysGreen'
)

register(
    id='MiniGrid-ColoredKeysBlue-v0',
    entry_point='gym_generalization.envs:ColoredKeysBlue'
)

register(
    id='MiniGrid-ColoredKeysPurple-v0',
    entry_point='gym_generalization.envs:ColoredKeysPurple'
)

register(
    id='MiniGrid-ColoredKeysGrey-v0',
    entry_point='gym_generalization.envs:ColoredKeysGrey'
)
