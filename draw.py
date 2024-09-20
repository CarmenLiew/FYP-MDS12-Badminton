import cv2

cap = cv2.VideoCapture('test.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
newVideo = cv2.VideoWriter('./pred_result/test_ball.mp4', fourcc, 30, (width, height))

with open('./pred_result/test_ball.csv', 'r') as f:
    lines = f.readlines()
    index = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        line = lines[index].split(',')
        x = int(float(line[2]))
        y = int(float(line[3]))
        if x != 0 and y != 0:
            cv2.circle(frame, (x,y), 10, (0, 255, 0), 2)
        newVideo.write(frame)
        index += 1
    # for index in range(1, len(lines)):
    #     line = lines[index].split(',')
    #     frame = int(line[0])
    #     x = int(float(line[1]))
    #     y = int(float(line[2]))


        # if frame % 10 == 0:
        #     print(frame)
        #     cap.set(1, frame)
        #     ret, img = cap.read()
        #     cv2.circle(img, (x, y), 10, (0, 255, 0), 2)
        #     cv2.imshow('img', img)
        #     cv2.waitKey(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break