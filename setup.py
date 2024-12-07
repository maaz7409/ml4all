from setuptools import setup, find_packages

setup(
    name="ml4all",                          
    version="0.1.0",                            
    author="maaz7409",
    description="",
    long_description=open("README.md").read(),  
    long_description_content_type="text/markdown", 
    # url="https://github.com/maaz7409/ml4all",  No need of this, as we have used labeled url
    packages=find_packages(), # automatically finds packages.                  
    classifiers=[
        "Programming Language :: Python 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",                   
    install_requires=[                         
        "numpy",
        "pandas",
        "matplotlib",
    ],
    project_urls={  # labeled links
        "GitHub": "https://github.com/maaz7409/ml4all",
    },
)
