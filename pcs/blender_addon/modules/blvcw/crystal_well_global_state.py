# GlobalState used for dynamic UI
import os

default_import_path = "pcs_crystals.obj"
# True if the provided object file could not be imported. Default True because initially no file is provided:
import_error = True
crystal_well_loader = None

default_output_directory = "output_dir"
# True if output_directory does not exist or could not be found:
output_directory_error = not os.path.exists(default_output_directory)
# True if the output_directory is not empty:
output_directory_warning = len(os.listdir(default_output_directory)) > 0 if not output_directory_error else False

# Will be set to True once a rendering has started. Prohibits new imports from that moment of time:
has_rendered = False
