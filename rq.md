this project in fake_quant is the implementation of resq (https://arxiv.org/abs/2412.14363), please understand the work firstly.
Then I want to control the rotation and quantization of ptq_model progress.
first, I want to  only rotate and quantize the module related with R1 (q_proj、k_proj、v_proj、o_proj, up_project, gate_project, and the right side of down_proj), I need to keep other module(the q,k,v related with R2,R3 in the attn computation， and left side of down project related with R4) in high bits without quantization.
second, I need a quanziation config file to control the quantization target optionally as above.
third, I need a modular port for the rotation and quantization for the left side of down_project(Custome R4).
help me to implement above requirement.
