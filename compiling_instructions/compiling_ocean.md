# To build/compile spectre on the ocean cluster @ CSUF:
1. assuming that you have a cloned updated version of spectre, if you don't or not sure follow step 1.1:
```
1.1. Update SpECTRE develop branch
1.2. cd into spectre dir and make sure you are on the main develop branch and then run the following cmds.
1.3. `$ git fetch upstream`
1.4. `$ git merge upstream/develop`
1.5. `$ git push origin develop`
1.6. Move onto step 2. 
```
2. `$ cd ~/spectre/build`
3. `$ export SPECTRE_HOME=/home/user_name/spectre`
4. `$ source ~/spectre/support/Environment/ocean_clang.sh`
5. `$ spectre_load_modules`
6. `$ spectre_run_cmake`
7. `$ make -j4`

# Testing executables / Ctests
* Testing all the SpECTRE executables
`$ make -j6  test-executables`
* github PR checks - shows the output when there are failures 
`$ ctest --output-on-failure -j<number of cores>`

# COMPILE AND RUN (location) an executable 
`$ make -j4 <Executable>`
`$ ./bin/<Executable>`
