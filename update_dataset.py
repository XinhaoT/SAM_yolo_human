import json
import os

def update_json(logfile_path):
    with open(logfile_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    
    img_info_list = content["images_list"]
    pic_idx = content["images_count"]
    
    ###initialize
    pic_idx = 0
    for img_info in img_info_list:
    	#TODO:
    	#Remove the infos which the corresponding pic are filtered
    
    for img_info in img_info_list:
    	old_idx = int(img_info["path"].split("/")[-1][:5])
    	#TO DO:
    	#Rename the source image
    	os.rename(img_info["path"], "dataset/used_raw_images/" + str(pic_idx).zfill(5) + ".jpg")
    	#Rename the cropped images
    	for i in range(len(img_info["sub_images"])):
    	    os.rename(img_info["sub_images"][i]["sub_image_path"], "dataset/processed_images/" + str(pic_idx).zfill(5) +"_"+str(i).zfill(2)+".jpg")
    	
    	img_info["path"] = "dataset/used_raw_images/" + str(pic_idx).zfill(5) + ".jpg"
    	for i in range(len(img_info["sub_images"])):
    	    sub_image = img_info["sub_images"][i]
    	    sub_image["sub_image_path"] = "dataset/processed_images/" + str(pic_idx).zfill(5) +"_"+str(i).zfill(2)+".jpg"
    	    
    	pic_idx += 1
    	
    

    
    new_content = {
        "images_count": pic_idx,
        "images_list": img_info_list
    }
    with open(opt.logfile_path, "w", encoding="utf-8") as f:
        json.dump(new_content, f, indent=4)
        
        
        
if __name__ == '__main__':
    update_json('dataset/datalog.json')
