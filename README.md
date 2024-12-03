# ABF processing (final clever name TBD)

This program uses wavelet decomposition of an EEG signal to detect anomalous activity and extract those anomalies. It allows for expert analysis and the streamlined creation of high quality datasets.

## Installation
This program requires python version 3.7 or higher installed on your machine.
It uses the included (windows users might have to check on that) [pip](https:pip.pypa.io/en/stable/) to install itself.
But fear not my non-programming friends, the included installation script should do the work for you.

The first step should be to open your favorite terminal on your preferred Operating System.
Open it inside the folder containing the installation script or access said folder via the "cd" (change directory) command.
```bash
cd [DIRECTORY PATH]
```
With [DIRECTORY PATH], the path from the current directory to the target directory.<br />
If you are unsure of your current whereabouts the "ls" command in linux and Mac OS lists the folder and files contained in the current directory, use the "dir" command to achieve the same in windows.

Most sensible OSes have restricted permissions on executable code so we will have to change those for the install script.<br />
On Mac OS and any flavor of Linux the following command should work fine:
```bash
sudo chmod 755 install.sh
```

Ok now we have permission to execute the script to install the necessary tools on your machine.
The install script creates a python virtual environment inside the program's folder structure, the "venv" folder contains that virtual environment.<br />
A virtual environment is a sandboxed version of python in which any libraries can be installed without affecting the main python version installed on your system. Simply deleting the "venv" folder or the entire folder containing this program will delete any installed tools and libraries.
Install the tool's dependencies as follows
```bash
./install.sh
```
Get yourself a coffee or a short break, just do it quickly this shouldn't take too long.
Now you should have a new "venv" folder inside the current directory.

## Usage
Using the tool requires activating the virtual environment and executing the main python file and then, when done, deactivating the virtual environment.<br />
To spare you the trauma of too much terminal tinkering the script "run" will take care of all that unpleasantness for you.
First you will need to change execution permissions of the script as previously.
```bash
sudo chmod 755 run.sh
```

And then just execute the script
```bash
./run.sh
```

If all the preceding steps didn't throw any errors or exceptions you should now be able to use the tool.

## Roadmap
For now the code allows accounts only for single channel files, if circumstances demands for multi-channel compatibility I'll add it.<br />
Current classifiers trained on lower quantity/quality datasets can't help accelerate the detection of full fledged seizures, their recall scores makes them useless. Future classifiers will be trained on datasets created with this very tools, when performance proves useful pretrained classifiers will be added to increase the rate of file analysis.<br />
I'll maybe allow for the use of different wavelets for the decomposition and extraction algorithm, I've seen different ones used in the litterature, more options is always better.

## Interface
The interface is meant to be the most explicit in its functionalities. Each interactive part of the interface has an associated shortcut, hovering the mouse cursor above a button or field displays a short description and the relevant shortcut at the bottom of the main window.

Before opening a file the main plot is available to get acquainted with the toolbar. The plot can be scaled and moved in both the X and Y axis. The transparent orange rectangle is a selection span that can be scaled and moved in the X axis.<br />
After opening the file the main plot scales the data to balance viewing comfort and amount of detail visible. In the event it fails at that task the user has complete control on the position, level of zoom on the signal, scale in both axes to facilitate analysis and subsequent selection of the part of the signal to extract and save.

By default selecting a save folder will allow the tool to save every anomalies and seizures in their respective folders from any abf files opened during the user's work session.
Select a save folder, open any number of abf files, extract seizure canditates, analyse the resulting anomalies, select the seizures, and click on the save button. All saved files will contain the name of the original files and their index in the list from which they were selected.