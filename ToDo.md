


- validate on get_frame() that images were actually got. 
  - if not then raise runtime error? 
  - return none or empty images/lists?
- add in the Frame object a Speeds object to track the perf times of all operations
  - getting the frame, rectifying, pre-inference-post, etc.