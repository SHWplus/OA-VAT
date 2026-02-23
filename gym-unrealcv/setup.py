from setuptools import setup, find_packages

setup(
    name='gym_unrealcv',
    version='1.0.0',
    install_requires=[
        'gym==0.10.9', 
        'matplotlib', 
        'numpy', 
        'unrealcv', 
        'wget', 
        'opencv-python'
    ],
    # 明确指定包含的包
    packages=find_packages(include=['gym_unrealcv', 'RandomRoom', 'gym_unrealcv.*', 'RandomRoom.*'])
)
