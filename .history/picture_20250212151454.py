def get_image_type(file_path):
    with open(file_path, 'rb') as f:
        file_header = f.read(10).hex().upper()
    
    if file_header.startswith('FFD8FF'):
        return 'JPEG'
    elif file_header.startswith('89504E47'):
        return 'PNG'
    elif file_header.startswith('47494638'):
        return 'GIF'
    elif file_header.startswith('424D'):
        return 'BMP'
    else:
        return 'Unknown'

# 示例用法
file_path = r"C:\Users\YukeZhou\Pictures\Camera Roll\ROG X EVANGELION.png"
image_type = get_image_type(file_path)
print(f'The image type is: {image_type}')
