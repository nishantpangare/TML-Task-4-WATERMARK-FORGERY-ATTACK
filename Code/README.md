1. We have two python files. 
    a. Extract_watermark.py
    b. task_template.py
2. Execute first file for extracting the watermark from the WM images. It uses the functions and the predefined model from the Github repo.
3. This code saves the deltas into a folder.
4. After that execute the task_template.py to execute the forgery attack and here we can load the delta folder and forge the target images with the extracted watermarks. 
5. Change alpha values to scale the residual. 
6. After executing the task_template.py it saves the zip file of 200 forged images which then can be submitted onto the server using the submission.py