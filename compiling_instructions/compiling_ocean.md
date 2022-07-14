# To build/compile spectre on the ocean cluster:
1. assuming that you have a cloned updated version of spectre, if you don't or not sure follow step 1.1:
1.1. 
# update spectre develop branch
```
1.2 cd into spectre dir and make sure you are on the main develop branch and then run the following cmds.
1.3 $ git fetch upstream
1.4 $ git merge upstream/develop
1.5 $ git push origin develop
```
2. $ cd ~/spectre/build
3. $ export SPECTRE_HOME=/home/user_name/spectre
4. $ source ~/spectre/support/Environment/ocean_clang.sh
5. $ spectre_load_modules
6. $ spectre_run_cmake
7. $ make -j4
