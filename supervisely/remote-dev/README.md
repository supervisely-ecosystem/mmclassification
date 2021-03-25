# Develop on remote machine

@TODO:
- change tutorial to root
- how to kill
- how to browse remote folder
- WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED ([link](https://www.digitalocean.com/community/questions/warning-remote-host-identification-has-changed))
- pycharm to browse files in container
- docker-compose-gpu
- mount directory for sources

You can debug your application using an interpreter that is located on the other computer, for example, on a web server or dedicated test machine. For example: you have laptop + server with GPU in the cloud and you want to develop apps to work with Neural Networks: train / inference / serve


## Configure SSH access to your remote server (optional one-time step)

### 1. Create an SSH Shortcut

If you are constantly needing to SSH into multiple servers, it can real daunting to remember all the different usernames, hostnames, IP addresses, and even sometimes custom private keys to connect to them. It's actually extremely easy to create command line shortcuts to solve this problem. More info here: https://linuxize.com/post/using-the-ssh-config-file/

Edit SSH config:
```sh
nano ~/.ssh/config
```

Add Server Info to the end of file:

```
Host gpu1
     HostName XX.XXX.XXX.XX
     User root
```

In this example shortcut is `gpu1`. `HostName` - is an ip-address to your server.

### 2. Set up public key authentication

ssh-copy-id installs an SSH key on a server as an authorized key. Its purpose is to provision access without requiring a password for each login. This facilitates automated, passwordless logins and single sign-on using the SSH protocol. More info here: https://www.ssh.com/ssh/copy-id


Check that you have SSH keys:
```sh
cat ~/.ssh/id_rsa.pub
```

If you see message `No such file or directory` then run following command to generate as SSH key:
```sh
ssh-keygen
```

Authorize you SSH key on remote server. I will need to enter remote server password
```
ssh-copy-id -i ~/.ssh/id_rsa.pub gpu1
```

### 3. Check SSH access

Now you can type `ssh gpu1` in terminal to connect to your remote server quickly.

<img src="https://i.imgur.com/8OZH2Xw.png"/>


## Run docker container on remote machine

### Prerequisites

Be sure, that [`docker`](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) and [`docker-compose`](https://docs.docker.com/compose/install/) are installed on your remote server.


### 1. Clone template repository

On remote server: create working directory (for example: `~/tutorial`) and clone template repository:

```sh
# run remote server
mkdir -p ~/tutorial && \
cd  ~/tutorial && \
git clone https://github.com/supervisely-ecosystem/remote-dev-template
```

### 2. Copy your SSH public key to template project folder 

Copy public SSH key from your local computer to remote server by executing following command:

Example: 
```sh
# run on local computer
scp ~/.ssh/id_rsa.pub <ssh shortcut>:<path to template project>/id_rsa.pub
```

For this tutorial we copy local file `~/.ssh/id_rsa.pub` to remote destination `~/tutorial/remote-dev-template/id_rsa.pub`; the command is the following:
```sh
# run on local computer
scp ~/.ssh/id_rsa.pub gpu1:~/tutorial/remote-dev-template/id_rsa.pub
```

### 3. Run docker container
In this section you will define base docker image. Then new docker image will be created on top of the defined one with 
installed and configured `openssh-server`.

backup link: https://www.dontpanicblog.co.uk/2018/11/30/ssh-into-a-docker-container/

Define your docker image in `docker-compose.yml` file. Then execute on a remote server:
```sh
# run on remote server
cd ~/tutorial/remote-dev-template && \
docker-compose up -d --build
```

Let's check the status of our started container:
```sh
# run on remote server
docker-compose ps
```

You should see something like this:
```
           Name                         Command               State          Ports
------------------------------------------------------------------------------------------
remotedevtemplate_devssh_1   sh -c /sshd_deamon.sh /ssh ...   Up      0.0.0.0:7777->22/tcp
```

FYI - to kill container:
```sh
# run on remote server
docker-compose kill
```

Now we have docker container running on remote server that can be directly accessed (securely) via SSH.  

### 4. SSH into container

Now you can SHH into container that is started on your remote server.

```sh
# run on local computer
ssh sshuser@gpu1 -p 7777
ssh root@gpu1 -p 7777
```

## TODO: Browse files on remote server
https://apple.stackexchange.com/questions/5209/how-can-i-mount-sftp-ssh-in-finder-on-os-x-snow-leopard

Install (MacOs):
```sh
brew install osxfuse
brew install sshfs
```

Mount remote directory  `/root/tutorial/remote-dev-template` to local directory `~/remote-dir`:
```sh
sshfs gpu1:/root/tutorial/remote-dev-template ~/remote-dir -ovolname=remote-dir
```

or (max)
```sh
sshfs gpu1:/root/work ~/gpu1-work -ovolname=gpu1-work
```


to unmount:
```sh
umount -f ~/remote-dir
```
or (max)
```sh
umount -f ~/gpu1-work
```


## Configure remote debug with PyCharm

With PyCharm you can debug your application using an interpreter that is located in docker container on remote server. 
This is a Professional feature: [download PyCharm Professional](https://www.jetbrains.com/pycharm/download) to try. 
Sources will be stored on your computer and automatically synchronized to remote container using SSH.

### 1. Clone repository with demo python project

```sh
# run on local computer
cd ~ && \
git clone https://github.com/supervisely-ecosystem/while-true-script && \
cd ~/while-true-script
```

### 2. Run PyCharm Professional

Press `Open` button:
<img src="https://i.imgur.com/aHZWKfn.png"/>

Go to demo project's folder `~/while-true-script` (1) and press `Open` button (2):
<img src="https://i.imgur.com/i4G9uvH.png"/>

### 3. Configure Project Interpreter

Wait untill PyCharm suggest to configure interpreter. Press `Cancel` button (1):
<img src="https://i.imgur.com/bdctiMg.png"/>

Open PyCharm preferences:
<img src="https://i.imgur.com/r3B4qef.png"/> 

Choose current project (1) and press `Configure`->`Add` button (2):
<img src="https://i.imgur.com/hbg3ec1.png"/> 

Let's add and configure remote python interpreter:

1. Choose `SSH interpreter`

In `New server configuration` section define:

2. remote host, in our example it is the SSH shortcut `gpu1`

3. remote port, we started container and shared its port 22 (ssh) to port 7777 on remote server

4. username inside container: `sshuser`

5. press `Next` button
<img src="https://i.imgur.com/ePu5OFl.png"/> 

Now PyCharm connected over SSH inside remote docker container.

Let's define path to python interpreter inside container. 
<img src="https://i.imgur.com/iI4YKp8.png"/> 

Python path for all Supervisely's docker images
is `/usr/local/bin/python3.8`
<img src="https://i.imgur.com/MPPu3RR.png"/>

Now remote python interpreter is defined, and it's time to define how sources from local computer will be synchronized 
to remote docker container.
<img src="https://i.imgur.com/G9JVeHh.png"/>

`Remote Path` - where the sources are stored inside remote docker container. Path has to start from `/home/sshuser/`. 
It is crucial because our user `sshuser` has write permissions only to this directory. In our example destination path 
is `/home/sshuser/tutorial-project`.
<img src="https://i.imgur.com/L80Si8Q.png"/>


Now everything is defines, press `Finish` button:
<img src="https://i.imgur.com/jq2vGYR.png"/>

You should see something like this - the list of python packages, press `OK` button:
<img src="https://i.imgur.com/PVD39Ft.png"/>

The last step is to install dependencies from `requirements.txt`:
<img src="https://i.imgur.com/EqDVNpO.png"/>


