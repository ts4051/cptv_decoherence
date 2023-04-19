'''
Recipe to install analysis software for CPT-V decoherence study
'''

import os, subprocess, collections, datetime, sys, argparse


#
# Tools
#

def check_output(cmd):
    '''
    A version of subprocess.check_output that can be used even where using versions of subprocess without that function
    Taken from https://stackoverflow.com/questions/26894024/subprocess-check-output-module-object-has-out-attribute-check-output/26894113#26894113
    '''
    process_list = []
    cmd_list = cmd.strip().split("|")
    for i, sub_cmd in enumerate(cmd_list):
        STDIN = None
        if i > 0:
            STDIN = process_list[i - 1].stdout
        process_list.append(subprocess.Popen(sub_cmd, stdin=STDIN, stdout=subprocess.PIPE, shell=True))
    if len(process_list) == 0:
        return ''
    output = process_list[i].communicate()[0]
    return output


def execute_commands(commands) :
    '''
    Run a list of shell commands in a single envionrment/shell
    Check output of each command before continuing
    '''

    assert isinstance(commands,list)

    overall_command = ""

    for command in commands :

        assert isinstance(command,str)

        if len(overall_command) > 0 : 
            overall_command += " ; "

        overall_command += command

        overall_command += " || exit 1" # Error code handling

    print(overall_command)

    return_code = subprocess.call( overall_command, shell=True, executable="/bin/bash" )
    assert return_code == 0



#
# Conda
#

def check_if_conda_env_exists(anaconda_bin, env_name) :
    '''
    Check if a conda env exists with the name specified
    '''

    # Get list of conda env names matching the one specified
    matching_conda_env = check_output( "%s/conda env list | awk '{ print $1; }' | grep -v '#' | grep '^%s$' ; exit 0" % (anaconda_bin,env_name) )
    
    # Check if found a match
    return len(matching_conda_env) > 0


def get_conda_env_init_commands(anaconda_bin, env_name=None) :
    '''
    Grab the commands required to init a conda env
    '''

    # Use absolute path
    anaconda_bin = os.path.abspath(anaconda_bin)
    assert os.path.isdir(anaconda_bin), "anaconda bin directory does not exist : %s" % anaconda_bin

    commands = []

    # Unset PYTHONPATH
    commands.append( "unset PYTHONPATH" )

    # Add anaconda to PATH
    commands.append( "export PATH=%s:$PATH" % anaconda_bin )

    # Activate conda env (if one is provided)
    if env_name is not None :
        commands.append( "source activate %s" % env_name )

    return commands


def create_conda_env(anaconda_bin, env_name, env_file=None, overwrite=False, conda_packages=None, pip_packages=None, python_version=None) :

    '''
    Create a conda env
    Return the commands required by a setup script to configure the environment to use this env
    '''

    # Check inputs
    assert os.path.isdir(anaconda_bin), "anaconda bin directory does not exist : %s" % anaconda_bin
    if env_file is not None :
        assert os.path.isfile(env_file), "env file does not exist : %s" % env_file

    # Need some handling to make the conda env play nicely with Jupyter notebooks
    if conda_packages is None :
        conda_packages = []
    if "ipykernel" not in conda_packages :
        conda_packages.append("ipykernel")

    # Init conda stuff
    conda_env_commands = get_conda_env_init_commands(anaconda_bin, env_name=env_name)

    # Check if conda env already exists
    conda_env_exists = check_if_conda_env_exists( anaconda_bin=anaconda_bin, env_name=env_name )

    # Decide whether to create a new conda env (either if doesn;t yet exist, or use wants to overwrite)
    need_to_create_env  = (not conda_env_exists) or overwrite
    if need_to_create_env :

        commands = []
        commands.extend(conda_env_commands[:-1]) # Ignore the "activate" command

        # Remove existing conda env
        # First call the conda function remove it
        # As a backup, also delete the directory (have seen cases where this was required, I think if the user does ctrl+c at a certain point)
        if conda_env_exists :
            conda_env_dir = os.path.join( anaconda_bin, "..", "envs", env_name)
            print(( "Removing existing conda env : %s (%s)" % (env_name,conda_env_dir) ))
            commands.append( "conda env remove -y -n %s ; rm -rf %s" % (env_name,conda_env_dir) )

        # Create the conda env
        # Note that command format is different if using an environment file
        #TODO support timeout? Used to need this at Madison but doesn't seem to anymore so probably can forget it.
        print("Creating conda env : %s" % env_name)
        if env_file is None :
            cmd = "conda create -y --name %s" % env_name
        else :
            cmd = "conda env create -y -n %s -f %s" % (env_name,env_file)
        if python_version is not None :
            cmd += " python=%s" % python_version
        if conda_packages is not None :
            cmd += " " + " ".join(conda_packages)
        commands.append(cmd)

        # Install any user-specified additional pip packages
        # Need to setup the env too
        if pip_packages is not None :
            conda_env_commands = get_conda_env_init_commands(anaconda_bin, env_name=env_name)
            commands.extend(conda_env_commands)
            commands.append( "pip install " + " ".join(pip_packages) )

        # Tell jupyter notebooks about this conda env
        commands.append("python -m ipykernel install --user --name %s" % env_name)

        execute_commands(commands)

    else :
        print( "conda env already exists (use `overwrite` arg if want to overwrite it) : %s" % env_name )

    # Done
    print(">>> Anaconda '%s' env creation complete!\n" % env_name)

    # Return commands required in a setup script
    return conda_env_commands



#
# Git
#

def clone_git_repo(repo_path, target_dir, branch=None, recursive=False, overwrite=False, git_protocol=None) :
    '''
    Clone a git repo
    '''

    # Check repo path
    #TODO

    # Handle git prototcol (e.g. ssh vs https) is one is specified
    if git_protocol is not None :
        if git_protocol.lower() == "https" :
            if repo_path.startswith("git") :
                repo_path = repo_path.replace("git@","https://").replace(".com:",".com/")
            assert repo_path.startswith("https")
        elif git_protocol.lower() == "ssh" :
            if repo_path.startswith("https") :
                repo_path = repo_path.replace("https://","git@").replace(".com/",".com:")
            assert repo_path.startswith("git")
        else :
            raise Exception( "Unknown git protocol '%s', choose between 'ssh' or 'https' (or None)" % git_protocol )

    print("Cloning git repo '%s" % repo_path)

    # Check if it already exists
    #TODO check the target dir is actually a git repo, not just a dir
    already_exists = os.path.exists(target_dir)

    # Safety checks
    assert target_dir not in ["/","."]

    # Check if clone is required
    do_clone = (not already_exists) or overwrite
    if do_clone :

        # Remove the existing clone if requires
        if already_exists :
            print(("Removing existing clone : %s" % target_dir))
            return_code = subprocess.call( "rm -rf %s" % target_dir , shell=True )
            assert return_code == 0, "Failed to remove existing clone : %s" % target_dir

        # Clone
        command = "git clone"
        if recursive :
            command += " --recursive"
        if branch is not None :
            command += " -b %s" % branch
        command += " %s %s" % (repo_path,target_dir)
        print(( "Clone git repo : %s" % command ))
        return_code = subprocess.call( command , shell=True )
        assert return_code == 0, "Failed to clone git repo"

    else :
        print(("Git repo clone already exists (use `overwrite` arg if want to overwrite it) : %s" % target_dir))


#
# Specific product installation recipes
#

def install_mceq(target_dir, repo_path=None, branch=None, overwrite=False, install_deps=True, anaconda_bin=None, env_name=None, git_protocol=None, classic_version=False ) :
    '''
    Install MCEq (https://github.com/afedynitch/MCEq)

    Note that MCEq changed significantly at some point, with the older version being referred to as the "classic" version.
    See https://github.com/afedynitch/MCEq_classic for details.
    Use the `classic_version` arg to get this older version. 
    '''


    #
    # Check inputs
    #

    # Absolute paths
    target_dir = os.path.abspath(target_dir)

    # If install dependencies, must provide conda env
    if install_deps :
        assert anaconda_bin is not None, "Must provide `anaconda_bin` when installing MCEq dependencies"
        assert env_name is not None, "Must provide `env_name` when installing MCEq dependencies"


    #
    # Handle current vs classic version
    #

    # Current version
    if not classic_version :

        # See https://mceq.readthedocs.io/en/latest/#installation

        #
        # Get MCEq
        #

        # Defaults
        if repo_path is None :
            repo_path = "git@github.com:afedynitch/MCEq.git"
        if branch is None :
            branch = "master"

        # Git clone
        clone_git_repo(
            repo_path=repo_path,
            target_dir=target_dir,
            branch=branch,
            recursive=False,
            overwrite=overwrite,
            git_protocol=git_protocol,
        )


        #
        # Install
        #

        install_commands = []

        # Prepare conda env
        conda_env_commands = get_conda_env_init_commands(anaconda_bin, env_name=env_name)
        install_commands.extend(conda_env_commands)

        # Install dependencies (defined in a file)
        requirements_file = os.path.join( target_dir, "requirements.txt" )
        assert os.path.isfile(requirements_file), "MCEq conda requirements files not found : %s" % requirements_file
        install_commands.append( "pip install -r " + requirements_file )

        # Install MCEq
        # Note that must be in the target directory for this to work
        install_commands.append("cd %s" % target_dir)
        install_commands.append( "python setup.py build_ext --inplace" )

        # Run the commands
        execute_commands(install_commands)

        # MCEq itself doesn't need installing as such, just needs the python path setting in any env scripts
        setup_commands = [
            "export MCEQ_DIR=" + target_dir,
            "export PYTHONPATH=$MCEQ_DIR:$PYTHONPATH",
        ]


    # Classic version
    else :

        #
        # Get MCEq
        #

        # Defaults
        if repo_path is None :
            repo_path = "git@github.com:afedynitch/MCEq_classic.git"
        if branch is None :
            branch = "master"

        # Git clone
        clone_git_repo(
            repo_path=repo_path,
            target_dir=target_dir,
            branch=branch,
            recursive=True,
            overwrite=overwrite,
            git_protocol=git_protocol,
        )


        #
        # Install dependencies
        #

        # Only if requested by user
        #TODO Current doesn't work on OSX (but everything required seems to be in anaconda anayway). Maybe just define them as alist instead...
        if install_deps : 

            install_commands = []

            # Prepare conda env
            conda_env_commands = get_conda_env_init_commands(anaconda_bin, env_name=env_name)
            install_commands.extend(conda_env_commands)

            # Install dependencies (defined in a file)
            conda_deps_file = os.path.join( target_dir, "conda_req.txt" )
            assert os.path.isfile(conda_deps_file), "MCEq conda dependencies files not found : %s" % conda_deps_file
            install_commands.append( "conda install -y -n " + env_name + " --file " + conda_deps_file )

            # Run the commands
            execute_commands(install_commands)


        #
        # MCEq environment
        #

        # MCEq itself doesn't need installing as such, just needs the python path setting in any env scripts
        setup_commands = [
            "export MCEQ_DIR=" + target_dir,
            "export PYTHONPATH=$MCEQ_DIR:$PYTHONPATH",
        ]


    #
    # Done
    #

    print(">>> MCEq installation complete!\n")

    return setup_commands


def install_nusquids(
    anaconda_bin, env_name,
    squids_target_dir, nusquids_target_dir,
    squids_repo_path=None, nusquids_repo_path=None, 
    squids_branch=None, nusquids_branch=None, 
    overwrite=False, 
    make=True, # Optionally build (nu)SQuIDS
    test=False, # Optionally test (nu)SQuIDS
    git_protocol=None,
) :
    '''
    Install nuSQuIDS (https://github.com/arguelles/nuSQuIDS)
    '''

    #
    # Check inputs
    #

    # Defaults
    if squids_repo_path is None :
        squids_repo_path = "git@github.com:jsalvado/SQuIDS.git"
    if squids_branch is None :
        squids_branch = "master"

    if nusquids_repo_path is None :
        nusquids_repo_path = "git@github.com:arguelles/nuSQuIDS.git"
    if nusquids_branch is None :
        nusquids_branch = "master"

    # Absoute paths
    squids_target_dir = os.path.abspath(squids_target_dir)
    nusquids_target_dir = os.path.abspath(nusquids_target_dir)


    #
    # Environment
    #

    # Define commands a user needs to setup an environment
    # Doing this now as need to use this during the build process

    nusquids_lib_path = os.path.join(squids_target_dir,"lib") + ":" + os.path.join(nusquids_target_dir,"lib") + ":$CONDA_PREFIX/lib"

    setup_commands = [
        "export SQUIDS_DIR=" + squids_target_dir, # path to SQuIDS
        "export NUSQUIDS_DIR=" + nusquids_target_dir, # path to SQuIDS
        "export LD_LIBRARY_PATH=" + nusquids_lib_path + ":$LD_LIBRARY_PATH", # Lib path
        "export PYTHONPATH=$NUSQUIDS_DIR/resources/python/bindings/:$PYTHONPATH", # Add python bindings
    ]


    #
    # Setup env
    #

    install_commands = []

    # Prepare conda env
    conda_env_commands = get_conda_env_init_commands(anaconda_bin, env_name=env_name)
    install_commands.extend(conda_env_commands)

    # Add the (nu)SQuIDS setup stuff to the commands
    install_commands.extend(setup_commands)


    #
    # Install dependencies
    #

    # Define (nu)SQuIDS dependencies
    dependencies = [
        "gsl",
        "hdf5",
        "py-boost", # Get boost along with this
    ]

    # Add to the comands
    install_commands.append( "conda install -y -n " + env_name + " -c anaconda " + " ".join(dependencies) )


    #
    # Install SQuIDS
    #

    # Git clone
    clone_git_repo(
        repo_path=squids_repo_path,
        target_dir=squids_target_dir,
        branch=squids_branch,
        recursive=False,
        overwrite=overwrite,
        git_protocol=git_protocol,
    ) 

    # Configure (point all dependency paths to the conda env)
    command = "cd " + squids_target_dir 
    command += " ; " + "./configure --with-gsl-incdir=$CONDA_PREFIX/include --with-gsl-libdir=$CONDA_PREFIX/lib"
    install_commands.append(command)

    # Make
    if make :
        install_commands.append("cd " + squids_target_dir + " ; make clean ; make")

    # Test (try building and running an example)
    if test :
        install_commands.append("cd " + os.path.join(squids_target_dir,"examples","VacuumNeutrinoOscillations") + " ; make clean ; make ; no | ./vacuum.exe")


    #
    # Install nuSQuIDS
    #

    # Git clone
    clone_git_repo(
        repo_path=nusquids_repo_path,
        target_dir=nusquids_target_dir,
        branch=nusquids_branch,
        recursive=False,
        overwrite=overwrite,
        git_protocol=git_protocol,
    )        

    # Configure (point all dependency paths to the conda env)
    command = "cd " + nusquids_target_dir 
    command += " ; " + "./configure --with-python-bindings --with-gsl=$CONDA_PREFIX --with-hdf5=$CONDA_PREFIX --with-boost=$CONDA_PREFIX --with-squids=" + squids_target_dir
    install_commands.append(command)

    # Make
    if make :
        install_commands.append( "cd " + nusquids_target_dir + " ; make clean ; make" )

    # Test (try running an example)
    if test :
        install_commands.append( "cd " + nusquids_target_dir + " ; make examples" ) # Make the examples
        install_commands.append( "cd " + nusquids_target_dir + "/examples/Single_energy ; ./single_energy" ) # Run an example

    # Make pybindings
    if make :
        install_commands.append( "cd " + os.path.join(nusquids_target_dir,"resources","python","src") + " ; make clean ; make" )

    # Test pybindings (check can import it)
    if test :
        install_commands.append( "python -c 'import nuSQUIDSpy'" )


    #
    # Done
    #

    # Finally, run the commands
    execute_commands(install_commands)

    print(">>> (nu)SQuIDS installation complete!\n")

    # Return the environment setup commands
    return setup_commands


def install_pisa(anaconda_bin, env_name, target_dir, repo_path=None, branch=None, overwrite=False, pisa_resources=None, git_protocol=None) :
    '''
    Install PISA (https://github.com/IceCubeOpenSource/pisa)
    '''

    #
    # Check inputs
    #

    # Defaults
    if repo_path is None :
        repo_path = "git@github.com:icecube/pisa.git"
    if branch is None :
        branch = "master"

    # Absolute paths
    target_dir = os.path.abspath(target_dir)

    # Check PISA resource format
    if pisa_resources is not None :
        assert isinstance(pisa_resources,collections.Sequence)


    #
    # Get PISA
    #

    # Git clone
    clone_git_repo(
        repo_path=repo_path,
        target_dir=target_dir,
        branch=branch,
        recursive=False,
        overwrite=overwrite,
        git_protocol=git_protocol,
    )


    #
    # Install PISA
    #

    print("Installing PISA")

    install_commands = []

    # Prepare conda env
    conda_env_commands = get_conda_env_init_commands(anaconda_bin, env_name=env_name)
    install_commands.extend(conda_env_commands)

    # Manually install `line_profiler` using conda, not PIP as the PISA installer tries to do (has errors)
    # This used to be required, but now the PISA install works without it and in fact it actually casues 
    # issues since it seems to force an update to python3.8 which is incompatible with PISA. So removing.
    # install_commands.append("conda install -y -n " + env_name + " line_profiler")

    # Install PISA
    pip_packages = [
        # "git+https://github.com/icecubeopensource/kde.git#egg=kde", # Manually specify the awkward KDE dependency (doesn't work as part of PISA's own installation) - Later note, this seems OK now so have removed
        "-e " + target_dir + "[develop]", # This is the main PISA installation
    ]
    install_commands.append("pip install " + " ".join(pip_packages) )

    # Run the commands
    execute_commands(install_commands)


    #
    # PISA environment
    #

    setup_commands = []

    # PISA path
    setup_commands.append( "export PISA="+target_dir )

    # PISA resources (take any input ones into account)
    if pisa_resources is None :
        pisa_resources = []
    pisa_resources.extend([
        "$PISA/pisa_examples/resources", # PISA locations
    ])
    setup_commands.extend([ "PISA_RESOURCES=%s:$PISA_RESOURCES" % res for res in pisa_resources ])
    setup_commands.append("export PISA_RESOURCES=$PISA_RESOURCES")

    # PISA cache
    setup_commands.append( "export PISA_CACHE=$PISA/../cache/" )

    # PISA floating point precision
    setup_commands.append( "export PISA_FTYPE=fp64" )

    # Some file systems (notably the Madison datastore) choke with HDF5 file locking issues. Circumvent this.
    setup_commands.append( "export HDF5_USE_FILE_LOCKING='FALSE'" )

    # Done
    print(">>> PISA installation complete!\n")
    return setup_commands



def install_deimos(target_dir, repo_path=None, branch=None, overwrite=False, install_deps=True, anaconda_bin=None, env_name=None, git_protocol=None, classic_version=False ) :
    '''
    Install DEIMOS (https://github.com/ts4051/deimos)
    '''


    #
    # Check inputs
    #

    # Absolute paths
    target_dir = os.path.abspath(target_dir)

    # If install dependencies, must provide conda env
    if install_deps :
        assert anaconda_bin is not None, "Must provide `anaconda_bin` when installing MCEq dependencies"
        assert env_name is not None, "Must provide `env_name` when installing MCEq dependencies"


    #
    # Get DEIMOS
    #

    # Defaults
    if repo_path is None :
        repo_path = "git@github.com:ts4051/deimos.git"
    if branch is None :
        branch = "main"

    # Git clone
    clone_git_repo(
        repo_path=repo_path,
        target_dir=target_dir,
        branch=branch,
        recursive=False,
        overwrite=overwrite,
        git_protocol=git_protocol,
    )


    #
    # Install
    #

    #TODO Add installer script in DEIMOS directly

    install_commands = []

    # Prepare conda env
    conda_env_commands = get_conda_env_init_commands(anaconda_bin, env_name=env_name)
    install_commands.extend(conda_env_commands)

    # Install dependencies
    pip_packages = [
        "odeintw",
    ]
    install_commands.append("pip install " + " ".join(pip_packages) )

    # Run the commands
    execute_commands(install_commands)

    # DEIMOS itself doesn't need installing as such, just needs the python path setting in any env scripts
    setup_commands = [
        "export DEIMOS_DIR=" + target_dir,
        "export PYTHONPATH=$DEIMOS_DIR:$PYTHONPATH",
    ]


    #
    # Done
    #

    print(">>> DEIMOS installation complete!\n")

    return setup_commands


#
# Setup script
#

def generate_setup_script(script_path, commands) :
    '''
    Function to generate a setup script
    Adds:
        1) Standard boiler plat
        2) Any commands the user defines
    '''

    # Check inputs
    assert isinstance(commands,collections.Sequence)

    # Start by adding bash shebang
    script_contents = '#!/bin/bash\n'

    # Unset python path
    script_contents += 'unset PYTHONPATH\n'

    # Magic line to get path to directory containing the script
    script_contents += 'THIS_DIR="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"\n'

    # Add all commands pass to this function
    for command in commands :
        script_contents += command + '\n'

    # Write the script
    with open(script_path,"w") as script :
        script.write(script_contents)
    print((">>> Setup script %s written\n" % script_path))



#
# Top-level installer
#

def install_analysis_software(
    install_dir, # Installation directory
    anaconda_bin, # Path to anaconda bin directory
    env_name=None, # Anaconda environment name
    env_file=None, # Anaconda environment file
    overwrite=False, # Can optionally overwrite any existing installation
    conda_env_kw=None, # keyword args to pass to `create_conda_env`
    pisa_kw=None, # keyword args to pass to `install_pisa`
    mceq=False, # Decide whether to install MCEq
    mceq_kw=None, # keyword args to pass to `install_mceq`
    nusquids=False, # Decide whether to install nuSQuIDS
    nusquids_kw=None, # keyword args to pass to `install_nusquids`
    deimos=False, # Decide whether to install DEIMOS
    deimos_kw=None, # keyword args to pass to `install_deimos`
    setup_commands=None, # Optionally user can provide a list of setup commands to add to the setup script
    git_protocol=None,
    setup_script_path=None, # Optionally can specify the path for the auto-generated setup script 
) :
    '''
    Install all software required for oscillations analyses.

    Put it all in a nice self-contained conda environment,
    including an auto-generated script for setting it up.

    Definitely includes:
        - PISA

    Optionally can include:
        - MCEq
        - nuSQuIDS
    '''

    start_time = datetime.datetime.now()


    #
    # Check inputs
    #

    # Get path to dir containing this file (used a few times later)
    this_dir_path = os.path.abspath(os.path.dirname(__file__))

    # Create installation dir if doesn't already exist
    assert isinstance(install_dir, str)
    if not os.path.isdir(install_dir) :
        os.makedirs(install_dir)

    # Defaults
    if env_name is None :
        env_name = "analysis"
    # if env_file is None :
    #     env_file = os.path.join( this_dir_path, "environment.yml" )
    if conda_env_kw is None :
        conda_env_kw = {}
    if mceq_kw is None :
        mceq_kw = {}
    if nusquids_kw is None :
        nusquids_kw = {}
    if pisa_kw is None :
        pisa_kw = {}
    if deimos_kw is None :
        deimos_kw = {}

    # Use absolute path for anaconda
    anaconda_bin = os.path.abspath(anaconda_bin)

    # Define target paths for each package
    # If provides one use that, otherwise create one
    if 'target_dir' not in pisa_kw.keys():
        pisa_kw['target_dir'] = os.path.join(install_dir,"pisa","src")

    if 'target_dir' not in mceq_kw.keys():
        mceq_kw['target_dir'] = os.path.join(install_dir,"MCEq","src")

    if 'squids_target_dir' not in nusquids_kw.keys():
        nusquids_kw['squids_target_dir'] = os.path.join(install_dir,"SQuIDS","src")

    if 'nusquids_target_dir' not in nusquids_kw.keys():
        nusquids_kw['nusquids_target_dir'] = os.path.join(install_dir,"nuSQuIDS","src")

    if 'target_dir' not in deimos_kw.keys():
        deimos_kw['target_dir'] = os.path.join(install_dir,"deimos","src")

    # Check setup commands format (should be a list of strings)
    if setup_commands is not None :
        assert isinstance(setup_commands,collections.Sequence)
        assert all([ isinstance(c,str) for c in setup_commands ])


    #
    # Install everything
    #

    # Create a conda env into which everything will be installed
    conda_env_setup_commands = create_conda_env(
        anaconda_bin=anaconda_bin,
        env_name=env_name,
        overwrite=overwrite,
        env_file=env_file,
        **conda_env_kw
    )

    # Install MCEq
    if mceq :
        mceq_setup_commands = install_mceq(
            anaconda_bin=anaconda_bin,
            env_name=env_name,
            overwrite=overwrite,
            git_protocol=git_protocol,
            **mceq_kw
        )

    # Install nuSQuIDS
    if nusquids :
        nusquids_setup_commands = install_nusquids(
            anaconda_bin=anaconda_bin,
            env_name=env_name,
            overwrite=overwrite,
            git_protocol=git_protocol,
            **nusquids_kw
        )

    # Install PISA
    pisa_setup_commands = install_pisa(
        anaconda_bin=anaconda_bin,
        env_name=env_name,
        overwrite=overwrite,
        git_protocol=git_protocol,
        **pisa_kw
    )

    # Install nuSQuIDS
    if deimos :
        deimos_setup_commands = install_deimos(
            anaconda_bin=anaconda_bin,
            env_name=env_name,
            overwrite=overwrite,
            git_protocol=git_protocol,
            **deimos_kw
        )

    #
    # Generate setup script
    #

    # Collect the setup commands from the various installation prodecures
    commands = []

    commands.append("\n# Conda env")
    commands.extend(conda_env_setup_commands)

    if mceq :
        commands.append("\n# MCEq env")
        commands.extend(mceq_setup_commands)

    if nusquids :
        commands.append("\n# (nu)SQuIDS env")
        commands.extend(nusquids_setup_commands)

    commands.append("\n# PISA env")
    commands.extend(pisa_setup_commands)

    if deimos :
        commands.append("\n# DEIMOS env")
        commands.extend(deimos_setup_commands)

    # Add any custom uder setup commands
    if setup_commands is not None :
        commands.extend(setup_commands)

    # Generate the script
    generate_setup_script(
        script_path=setup_script_path,
        commands=commands,
    )


    #
    # Done
    #

    end_time = datetime.datetime.now()
    time_take = end_time - start_time

    print(( "\n>>> Done! Took %s" % time_take ))
    print(( ">>> To setup the environment, use the following command at each new shell session : source <path/to/>%s" % os.path.basename(setup_script_path) ))



#
# Main
#

if __name__ == "__main__" :

    # Get the args
    parser = argparse.ArgumentParser()
    parser.add_argument('-ab','--anaconda-bin', type=str, required=True, help='anaconda bin directory' )
    parser.add_argument('-ow','--overwrite', action="store_true", required=False, help='overwrite any existing installation' )
    args = parser.parse_args()

    # Get path to this repo
    this_repo_dir = os.path.abspath( os.path.join( os.path.dirname(__file__), ".." ) )

    # Define python version
    python_version = "3.7.5"

    # Get install dir
    install_dir = os.path.join(this_repo_dir, "ext")
    print("Install dir : %s" % install_dir)

    # Get anaconda bin directory
    anaconda_bin = args.anaconda_bin
    while anaconda_bin.endswith("/") :
        anaconda_bin = anaconda_bin[:-1]
    if not anaconda_bin.endswith("bin") :
        os.path.join(anaconda_bin, "bin")
    anaconda_bin = os.path.abspath(anaconda_bin)
    print("Anaconda bin : %s" % anaconda_bin)

    # Environment name
    env_name = "cptv_decoh"

    # Define any additional packages to install
    conda_packages = [
        # "llvmlite==0.30.0", # segfaults with 0.31.0 #TODO Think this is no longer required as a hack
    ]
    if "darwin" in sys.platform.lower() : # OS X
        conda_packages.extend(["clang","clangxx",])
    # # else : :
    # #     conda_packages.extend(["gcc_linux-64","gxx_linux-64",])

    # Also packages that must be installed via pip, not cnda
    pip_packages = [
        "uncertainties", # PISA doesn't seem to properly install `uncertainties`, add it here
        "odeintw", # Needed for DensityMatrixOscSolver
    ]

    # Steer conda env
    conda_env_kw = {
        "conda_packages" : conda_packages,
        "pip_packages" : pip_packages,
        "python_version" : python_version,
    }

    # Steer PISA installation
    pisa_kw = {
        "repo_path" : "git@github.com:ts4051/pisa.git",
        "branch" : "master", 
        "pisa_resources" : [ os.path.join( this_repo_dir )], # Use resources in this repo
    }

    # Steer MCEq installation
    mceq_kw = {
        "classic_version" : False
    }

    # Steer nuSQuIDS installation
    nusquids_kw = {
        "squids_repo_path" : "git@github.com:ts4051/SQuIDS.git",
        "squids_branch" : "decoherence_py3",
        "nusquids_repo_path" : "git@github.com:ts4051/nuSQuIDS.git",
        "nusquids_branch" : "decoherence_py3",
        "make" : True,
        "test" : False,
    }

    # Add this repo to python path
    print("This repo dir : %s" % this_repo_dir)
    setup_commands = [
        "export CPTV_DECOH_DIR=" + this_repo_dir,
        "export PYTHONPATH=$CPTV_DECOH_DIR:$PYTHONPATH",
    ]

    # Setup script path
    setup_script_path =  os.path.join( this_repo_dir, "setup.sh" )

    # Run installer
    install_analysis_software(
        install_dir=install_dir,
        anaconda_bin=anaconda_bin,
        env_name=env_name,
        conda_env_kw=conda_env_kw,
        overwrite=args.overwrite,
        pisa_kw=pisa_kw,
        mceq=True,
        mceq_kw=mceq_kw,
        nusquids=True,
        nusquids_kw=nusquids_kw,
        deimos=True,
        setup_commands=setup_commands,
        setup_script_path=setup_script_path,
    )
