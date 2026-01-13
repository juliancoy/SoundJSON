import os
import sys
import json

def toDict(directory):
    retdict = {}
    length = 0

    for file in os.listdir(directory):
        fullfilename = os.path.join(directory, file)
        print(fullfilename)

        if os.path.isdir(fullfilename):
            l, fileresult = toDict(fullfilename)
            length += l
            if l:
                retdict[file] = fileresult
        else:
            length += 1
            retdict[file[:-4]] = "/" + fullfilename

    return length, retdict


# Use CLI argument if provided, otherwise default
directory = sys.argv[1] if len(sys.argv) > 1 else "samples"

length, dd = toDict(directory)

# Output file goes INSIDE the target directory
output_path = os.path.join(directory, "sitemap.json")

with open(output_path, "w+") as f:
    json.dump(dd, f, indent=2)

print(f"Wrote sitemap to {output_path}")

