from typing import Dict
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class NoopImageRunner(BaseImageRunner):
    def __init__(self, output_dir: str = '.'):
        super().__init__(output_dir)

    def run(self, policy) -> Dict:
        # Do nothing and return empty metrics
        return {} 