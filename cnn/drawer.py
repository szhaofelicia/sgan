import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch


class TrajectoryDrawer:
    field_width = 28.7
    field_height = 15.2

    def __init__(self, sampling_scale_factor=50, target_size=[572, 572], drawing_mode="blur"):
        self.sampling_scale_factor = sampling_scale_factor
        self.drawing_mode = drawing_mode
        self.target_size = target_size
        self.agent_num = 11
        self.sampling_resolution = (int(sampling_scale_factor * TrajectoryDrawer.field_width),
                                    int(sampling_scale_factor * TrajectoryDrawer.field_height))

    def transform_sampling_coords(self, x, y):
        return x * self.sampling_scale_factor, y * self.sampling_scale_factor

    def generate_trajectory_image(self, agent, image):
        for i in range(agent.size(0)):
            x = agent[i, 0].item()
            y = agent[i, 1].item()
            x, y = self.transform_sampling_coords(x, y)
            center_coordinates = (int(x), int(y))
            radius = 3
            color = 128
            thickness = -1
            image = cv2.circle(image, center_coordinates, radius, color, thickness)

            image = cv2.GaussianBlur(image, (5, 5), 0)
            image = cv2.GaussianBlur(image, (5, 5), 0)
            image = cv2.GaussianBlur(image, (5, 5), 0)
            image = cv2.GaussianBlur(image, (5, 5), 0)
            image = cv2.GaussianBlur(image, (5, 5), 0)
            image = cv2.GaussianBlur(image, (5, 5), 0)
            kernel = np.ones((3, 3), np.uint8)
            resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)

        return resized

    def generate_scene_image(self, scene):
        channels = []
        for i in range(scene.size(1)):
            agent_image = np.zeros((self.sampling_resolution[1], self.sampling_resolution[0]), np.uint8)
            agent = scene[:, i, :]
            channel = self.generate_trajectory_image(agent, agent_image)
            channel = np.array(channel)
            channels.append(channel)
        scene_image = np.stack(channels, axis=2)
        return scene_image

    def generate_batch_images(self, traj_batch):
        batch_size = traj_batch.size(1) // self.agent_num
        batch_images = []
        for i in range(batch_size):
            scene = traj_batch[:, i * self.agent_num: (i + 1) * self.agent_num, :]
            scene_image = self.generate_scene_image(scene)
            batch_images.append(scene_image)
        batch_images = np.stack(batch_images)
        batch_images = torch.Tensor(batch_images)
        batch_images = batch_images.permute(0, 3, 1, 2)
        return batch_images

    def generate_channel(self, agent):
        agent_image = np.zeros((self.sampling_resolution[1], self.sampling_resolution[0]), np.uint8)
        print(agent.size())
        channel = self.generate_trajectory_image(agent, agent_image)
        channel = np.array(channel)
        return channel