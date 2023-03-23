import mujoco as mj
import numpy as np
from mujoco.glfw import glfw


from mujoco_base import MuJoCoBase

DOWN = 0
UP = 1


class Projectile(MuJoCoBase):
    def __init__(self, xml_path):
        super().__init__(xml_path)

    def reset(self):
        # Set initial position of ball
        self.data.qpos[2] = 0.1

        # Set initial velocity of ball
        self.data.qvel[2] = 1

        # Set camera configuration
        self.cam.azimuth = 90.0
        self.cam.distance = 8.0
        self.cam.elevation = -45.0
        
        self.state = 1

        mj.set_mjcb_control(self.controller)

    def controller(self, model, data):
        
        if self.data.qpos[2] <= 0: self.state += 1; self.state %= 2
        if self.data.qpos[2] >= 2: self.state +=1; self.state %= 2
        
        if self.state == UP: self.data.qvel[2] = 9.8
        elif self.state == DOWN: self.data.qvel[2] = -9.8
        
    def simulate(self):
        while not glfw.window_should_close(self.window):
            simstart = self.data.time

            while (self.data.time - simstart < 1.0/60.0):
                # Step simulation environment
                mj.mj_step(self.model, self.data)

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Make camera track ball
            self.cam.lookat[0] = self.data.qpos[0]

            # Show world frame
            self.opt.frame = mj.mjtFrame.mjFRAME_WORLD

            # Update scene and render
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()


def main():
    xml_path = "./model/ball/ball.xml"
    sim = Projectile(xml_path)
    sim.reset()
    sim.simulate()


if __name__ == "__main__":
    main()