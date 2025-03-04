import os
import sys

if __name__ == "__main__":
    # set the environment variable
    HD_DATA_NAME = ""
    LD_DATA_NAME = ""
    if len(sys.argv) == 3:
        if os.path.isfile("../DATA/"+sys.argv[1]+".csv") and os.path.isfile("../DATA/"+sys.argv[2]+".csv"):
            HD_DATA_NAME = sys.argv[1]
            LD_DATA_NAME = sys.argv[2]
        else:
            print("File not found")
            sys.exit(1)
    if len(sys.argv) == 2:
        if os.path.isfile("../DATA/"+sys.argv[1] + "HD.csv") and os.path.isfile("../DATA/"+sys.argv[1] + "LD.csv"):
            HD_DATA_NAME = sys.argv[1]+"HD"
            LD_DATA_NAME = sys.argv[1]+"LD"
        else:
            print("File not found")
            sys.exit(1)
    with open(".env", "w") as f:
        f.write(f"HD_DATA_NAME={HD_DATA_NAME}\n")
        f.write(f"LD_DATA_NAME={LD_DATA_NAME}\n")
