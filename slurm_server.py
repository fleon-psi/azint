from subprocess import Popen, PIPE
from flask import Flask, request

app = Flask(__name__)


@app.route('/', methods=["POST"])
def hello():
    config = request.json

    text = ("#!/bin/bash \n"
            "#SBATCH --job-name=azint \n"
            "#SBATCH --cpus-per-task=32 \n"
            "#========================================\n"

            "# Load modules \n"
            "source /opt/gfa/python 3.8\n"
            "source activate /sls/MX/applications/conda_envs/azint\n"
            "python process.py {} {} {} {} {} {} {}\n").format(config["file"],
                                                               config["poni"],
                                                               config["mask"],
                                                               int(config["x0"]),
                                                               int(config["x1"]),
                                                               int(config["y0"]),
                                                               int(config["y1"]))

    p = Popen(['sbatch'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    p.communicate(input=text.encode())
