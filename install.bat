echo "The install script is executing in the project directory to install a python virtual environment"

python -m pip install --user --upgrade pip
if %errorlevel% == 0 (
    echo "\n\nHello my biologist friends, it\'s me your friendly neighbourhood programmer"
    echo "python 3 is installed on the machine, this script will install a sandboxed python environment with all the relevant libraries\n"
    echo "A new directory will be created at the root directory of the project, do not be alarmed, it is expected\n"
    echo "The new venv directory will contain all the installed libraries and will not encumber your system level installation of python\n"
    echo "If you wish to erase all traces of this tool and the associated python libraries, a simple deletion of the project directory will do just that\n"
    echo "Have fun!\n"
    
    python -m pip install --user virtualenv
    python -m venv venv
    .\venv\Scripts\activate && pip install -r requirements.txt && deactivate
)
else(
    ::echo $'\U0001f972'
    echo "\n\nThe script failed to detect python 3, make sure python version 3.7 or above is installed on your system\n\n"
)
