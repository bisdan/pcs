#!/usr/bin/env python
"""
Execute headless execution via a provided vcw_settings.json file.
Call via:
python3 headless_execution.py your_settings_file.json
"""

import os
import argparse

BLENDER_PATH="blender" # Change to your blender executable
def main(settings_file):
    script_path = "modules/blvcw/crystal_well_headless.py"
    blender_path = BLENDER_PATH
    blender_path += " --background"
    blender_path += " --python " + script_path
    blender_path += " --settings_file " + settings_file
    os.system(blender_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', default="", type=str,
                        help="Path to settings file used to render.")

    args = parser.parse_args()
    main(args.settings_file)
