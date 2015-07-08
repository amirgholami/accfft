/**
 * \file mat_utils.txx
 * \author Dhairya Malhotra, dhairya.malhotra88@gmail.com
 * \date 1-1-2013
 */
#include <cstring>
#include <stdlib.h>
#include <iostream>

const char* commandline_option(int argc, char** argv, const char* opt, const char* def_val, bool required, const char* err_msg){
  for(int i=0;i<argc;i++){
    if(!strcmp(argv[i],opt)) return argv[i+1];
  }
  if(required){
    std::cout<<err_msg<<'\n';
    exit(0);
  }
  return def_val;
}

