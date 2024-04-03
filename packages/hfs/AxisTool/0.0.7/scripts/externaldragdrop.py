from drop import dragdrop

reload(dragdrop)

def dropAccept(file_list):
    dragdrop.dropAccept(file_list)

    return True