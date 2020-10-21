#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os
import glob
import pathlib
import shutil

def show1(img):
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def show2(img1,img2):
    cv2.imshow("img1",img1)
    cv2.imshow("img2",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize(img, percent=0.2):
    height = img.shape[0]
    width  = img.shape[1]
    img_resize = cv2.resize(img , (int(width*percent), int(height*percent)))
    return img_resize
    
#ディレクトリ内のファイル全てを読み込む関数
def load_file(folder, fmt="png"):
    images = []
    files = sorted(glob.glob(folder + '/*.' + fmt))
    #print(files)
    if fmt is "png":
        for filename in files:
            #print(filename)
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img) 
        print("{} is loaded.\n".format(folder))
        return images
    else:
        for filename in files:
            if filename is not None:
                images.append(filename)
        print("{} is loaded.\n".format(folder))
        return images
    
#画像をリストで渡すと横向きに連結してくれる関数(サイズに要注意)
def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)
    
#画像を横に分割する関数
def devide_image(img, width, trimming_point,bias=0):
    devide_width = width*trimming_point
    devide_width = int(devide_width+0.5)

    left_img = img[:,0:devide_width+bias]
    right_im = img[:,devide_width-bias:width]
    
    return left_img, right_im

#transformECCによる位置合わせ関数
def align(base_img, target_img, warp_mode=cv2.MOTION_TRANSLATION, number_of_iterations=5000, termination_eps=1e-10):
    #base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    #target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    base_gray = base_img
    target_gray = target_img

    # prepare transformation matrix
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    sz = base_img.shape

    # estimate transformation
    try:
        (cc, warp_matrix) = cv2.findTransformECC(base_gray, target_gray, warp_matrix, warp_mode, criteria)

        # execute transform
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography 
            aligned = cv2.warpPerspective(target_img, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            aligned = cv2.warpAffine(target_img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        return aligned
    except Exception as ex:
        print("can not align the image")
        return target_img

#スティッチング関数
def stitcing(img1,img2):
    #img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    #img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # 特徴点 Key Points kp1, kp2
    # 特徴量記述子 Feature Description des1, des2
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    # 特徴量を総当たりでマッチングします。
    # マッチング度合いが高い順に二つ (k=2) 取得します。
    match = cv2.BFMatcher()
    matches = match.knnMatch(des2, des1, k=2)

    # マッチング結果に閾値を設定します。
    # 取得した結果二つのうち、一つをもう一つの閾値として利用しています。
    good = []
    for m, n in matches:
        if m.distance < 0.03 * n.distance:
        # if m.distance < 0.75 * n.distance:
            good.append(m)

    # ホモグラフィの計算には理論上 4 つの点が必要です。実際にはノイズの影響もあるため更に必要です。
    MIN_MATCH_COUNT = 5
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp2[m.queryIdx].pt for m in good ])
        dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good ])
        H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]
    else:
        print('Not enought matches are found - {}/{}'.format(len(good), MIN_MATCH_COUNT))
        exit(1)

    # ホモグラフィ行列で img2 を変換します。
    img2_warped = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))

    # img1 と結合します。
    img_stitched = img2_warped.copy()
    img_stitched[:img1.shape[0], :img1.shape[1]] = img1
    draw_params = dict(matchColor=(0,255,0),
                   singlePointColor=None,
                   flags=2)
    save("hoge",cv2.drawMatches(img2, kp2, img1, kp1, good, None, **draw_params))

    # 余分な 0 領域を削除します。
    def trim(frame):
        if np.sum(frame[0]) == 0:
            return trim(frame[1:])
        if np.sum(frame[-1]) == 0:
            return trim(frame[:-2])
        if np.sum(frame[:,0]) == 0:
            return trim(frame[:, 1:])
        if np.sum(frame[:,-1]) == 0:
            return trim(frame[:, :-2])
        return frame
    img_stitched_trimmed = trim(img_stitched)
    
    return img_stitched_trimmed

#cv画像配列を引数とした最大外接矩形の４隅の座標を返す関数(前処理はやってね)
def get_corner_position(img):
    contours = cv2.findContours(img,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    rect=[]
    area=[]
    for i,cnt in enumerate(contours):
        rect.append(cv2.minAreaRect(cnt))
        w,h=rect[i][1]
        area.append(w*h) 
    
    #最大面積の矩形だけをとる
    max_area = max(area)
    max_index = area.index(max_area)
    max_rect = rect[max_index]
    #print(max_rect)
    box = cv2.boxPoints(max_rect)
    box = np.int0(box)
    
    return box,contours

#座標を与えるとその座標に画像を変換してくれる関数
def convert_img_byUsingCorner(ref_pts, tgt_pts,tgt_img):
    #[[左上],[左下],[右下],[右上]]
    ref_pts = np.float32(ref_pts)
    tgt_pts = np.float32(tgt_pts)
    h,w = tgt_img.shape[:2]

    M = cv2.getPerspectiveTransform(ref_pts,tgt_pts)
    dst = cv2.warpPerspective(tgt_img,M,(w,h))
    
    return dst

#reference_imageのコーナー座標にtarget_imageのコーナーを合わせる関数
def homuhomu(img_ref, img_tgt):
    kernel_3 = np.ones((3,3),np.uint8)
    kernel_5 = np.ones((5,5),np.uint8)
    kernel_7 = np.ones((7,7),np.uint8)
    
    th_ref = cv2.threshold(img_ref, 80, 255, 0)[1]
    th_tgt = cv2.threshold(img_tgt, 80, 255, 0)[1]
    dilation_ref = cv2.dilate(th_ref,kernel_5,iterations = 1)
    dilation_tgt = cv2.dilate(th_tgt,kernel_5,iterations = 1)

    ref_corner,_ = get_corner_position(dilation_ref)
    tgt_corner,contours = get_corner_position(dilation_tgt)
    
    #全てのコーナー書きたくなったらここ
    #for i,cnt in enumerate(contours):
        #img2 = cv2.drawContours(img_tgt, [cnt], -1, (0,255,0), 3)
    
    #4隅のコーナー書きたくなったらここ
    #img1 = cv2.drawContours(img_ref, [ref_corner], -1, (0,0,255), 3)
    #img2 = cv2.drawContours(img_tgt, [tgt_corner], -1, (0,0,255), 3)
    #show2(img1,img2)
    dst = convert_img_byUsingCorner(ref_corner,tgt_corner,img_tgt)
    
    #パディングした分元に戻す
    dst = dst[:,:-1]

    return dst

#左サイドの座標使いたいからそこを返す関数(×)
#画像の前処理をしてコーナーをとってくれる関数に投げるだけに成り下がった名前詐欺関数
#本線と依存関係のため消すのめんどくさい。スーパーリファクタリングターゲット
def get_leftside_points(img_ref):
    kernel_3 = np.ones((3,3),np.uint8)
    kernel_5 = np.ones((5,5),np.uint8)
    kernel_7 = np.ones((7,7),np.uint8)
    kernel_9 = np.ones((9,9),np.uint8)
    
    th_ref = cv2.threshold(img_ref, 80, 255, 0)[1]
    #dilation_ref = cv2.dilate(th_ref,kernel_5,iterations = 1)
    erosion = cv2.erode(th_ref,kernel_5,iterations = 1)
    p_dilation = cv2.dilate(erosion,kernel_7,iterations = 1)
    #for i in range(5):
    #p_dilation = differential(p_dilation)
        #print(i)
    p_dilation = cv2.dilate(erosion,kernel_9,iterations = 1)
    #show1(p_dilation)
    cv2.imwrite( './p_dilation.png', p_dilation)

    ref_corner,_ = get_corner_position(p_dilation)
    #img1 = cv2.drawContours(img_ref, [ref_corner], -1, (255,255,255), 15)
    #cv2.imwrite( './test.png', img1)
    

    return ref_corner

#与えられたカーネルで計算する関数
def filter2d(src, kernel):
    # カーネルサイズ
    m, n = kernel.shape

    # 畳み込み演算をしない領域の幅
    d = int((m-1)/2)
    h, w = src.shape[0], src.shape[1]

    # 出力画像用の配列（要素は全て0）
    dst = np.zeros((h, w))

    for y in range(d, h - d):
        for x in range(d, w - d):
            # 畳み込み演算
            dst[y][x] = np.sum(src[y-d:y+d+1, x-d:x+d+1]*kernel)

    return dst

#画像の上部分とした部分の傾きを求める関数
def differential(img):
    #img = np.array(img, dtype=np.float64)
    #水平方向微分カーネル
    kernel_x = np.array([[-1.0, 0.0, 1.0],
                     [-2.0, 0.0, 2.0],
                     [-1.0, 0.0, 1.0]])
    #垂直方向微分カーネル
    kernel_y = np.array([[-1.0, -2.0, -1.0],
                     [0.0, 0.0, 0.0],
                     [1.0, 2.0, 1.0]])
    #平滑化カーネル
    kernel_n = np.array([[0.1, 0.1, 0.1],
                     [0.1, 0.1, 0.1],
                     [0.1, 0.1, 0.1]])
    
    #左上鋭角化カーネル
    kernel_ori = np.array([[0, 0, 1],
                     [0, 0, 0],
                     [15, 10, 0]])
    
    th = cv2.threshold(img, 80, 255, 0)[1]
    dst_x = filter2d(img, kernel_ori)
    dst_y = filter2d(img, kernel_y)
    
    dst_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    dst_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    
    dst = np.sqrt(dst_x ** 2 + dst_y ** 2)
    
    #show2(dst_x,dst_y)
    #show1(dst)
    
    return dst_x
    
#画像内四角形の上底と下底の傾きをそれぞれ計算し、返す関数
def get_slope(img):
    kernel_3 = np.ones((3,3),np.uint8)
    kernel_5 = np.ones((5,5),np.uint8)
    kernel_7 = np.ones((7,7),np.uint8)
    kernel_9 = np.ones((9,9),np.uint8)
    
    th = cv2.threshold(img, 80, 255, 0)[1]
    erosion = cv2.erode(th,kernel_3,iterations = 1)
    p_dilation = cv2.dilate(erosion,kernel_7,iterations = 1)
    p_dilation = cv2.dilate(p_dilation,kernel_9,iterations = 1)
    
    
    #両側が境界と引っ付いているのでパディングして輪郭を抽出できるようにする
    dilation = np.pad(p_dilation,(1,1),"constant")
    
    #show1(dilation)
    #cv2.imwrite( './test.png', dilation)

    corner,contours = get_corner_position(dilation)
    
    #上でパディングした分トリミングしてサイズを元に戻す
    dilation = dilation[1:-1,1:-1]
    
    dilation = cv2.drawContours(dilation, [corner], -1, (0,0,122), 30)
    
    #show1(dilation)
    #corner=[左下][左上][右上][右下]
    #print(corner)
    
    upper_slope = (float(corner[2][1])-float(corner[1][1])) / (float(corner[2][0])-float(corner[1][0]))
    lower_slope = (float(corner[3][1])-float(corner[0][1])) / (float(corner[3][0])-float(corner[0][0]))
    upper_intercept = float(corner[1][1])
    lower_intercept = float(corner[0][1])
    
    #print(corner[1][1],corner[0][1])
    #print(upper_slope, lower_slope, upper_intercept, lower_intercept)
    
    
    return upper_slope, lower_slope, upper_intercept, lower_intercept

#傾きから左側画像を補正する関数
def homu_with_slope(left_img,right_img,u_slope,l_slope,u_intercept,l_intercept,left_img_corner):
    kernel_3 = np.ones((3,3),np.uint8)
    kernel_5 = np.ones((5,5),np.uint8)
    kernel_7 = np.ones((7,7),np.uint8)
    kernel_9 = np.ones((9,9),np.uint8)

    x_intercept = left_img.shape[1]
    
    #左サイドのx座標と右サイドと左サイドのx座標の差が必要。
    #下に向かって正のグラフだから傾きに正をかけている。
    u_distance_x = left_img_corner[2][0] - left_img_corner[1][0]
    l_distance_x = left_img_corner[3][0] - left_img_corner[0][0]
    correct_uy = ((x_intercept-u_distance_x)*(-1)*u_slope)+u_intercept
    correct_ly = ((x_intercept-l_distance_x)*(-1)*l_slope)+l_intercept
    
    #print(u_intercept,l_intercept)
    
    #img1 = cv2.drawContours(left_img_ref, [im_corner], -1, (0,0,255), 8)
    #show1(img1)
    #print("hoge",correct_uy,correct_ly)
    
    #[[左上],[左下],[右下],[右上]]
    ref_pts = np.float32([[left_img_corner[1][0], left_img_corner[1][1]],
                       [left_img_corner[0][0], left_img_corner[0][1]],
                       [x_intercept, l_intercept],
                       [x_intercept, u_intercept]])
    #分断場所がx=0で左と下に向かって正のグラフだからx_interceptからマイナスしている
    tgt_pts = np.float32([[x_intercept-u_distance_x, correct_uy],
                       [x_intercept-l_distance_x, correct_ly],
                       [x_intercept, l_intercept],
                       [x_intercept, u_intercept]])
    
    #print(ref_pts,tgt_pts)
    
    correct_img = convert_img_byUsingCorner(ref_pts, tgt_pts,left_img)
    
    #show2(correct_img,left_img)
    
    return correct_img

#画像を左右に分割して右側画像から上底と下底の勾配を求め、その勾配を利用して左側画像の歪みを補正をする関数
def make_correct_img_leniarly(img_ref, devide_rate):
    #img_ref = cv2.imread("./test/test-ok/image009.png")
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    #img_ref = resize(img_ref)

    height,width = img_ref.shape[:2]
    bias = 150

    #引数(画像、その画像の横幅、分割したい割合、分割したい個所から指定ピクセル多く作る(初期値は0))
    #スティッチングするときにbias分の余裕がないと特徴点が一致しなかったのだが、結局スティッチングしなくなったから要らぬ引数
    left_img_ref, right_img_ref = devide_image(img_ref,width,devide_rate,bias=0)

    #上底と下底の傾きと切片を求める
    upper_slope, lower_slope,upper_intercept,lower_intercept = get_slope(right_img_ref)

    #傾きと切片から左側の画像の歪みをなくし、長方形になるようにする
    correct_img = homu_with_slope(left_img_ref,right_img_ref,upper_slope, lower_slope,upper_intercept,lower_intercept)

    fusion_img = hconcat_resize_min([correct_img,right_img_ref])

    #left_fusion_img,right_fusion_img = devide_image(fusion_img,width,devide_rate,bias)

    #show2(fusion_img,right_fusion_img)
    #stitcing_fusion_img = stitcing(left_fusion_img,right_fusion_img)
    
    return fusion_img

#make_correct_img_leniarlyの色々なデータを返してくれるVer
def make_correct_img_leniarly_returnData(img_ref, devide_rate):
    height,width = img_ref.shape[:2]
    
    #スティッチングする場合、分割画像に少しゆとりを持たせないと共通の特徴点がなくなる
    #biasの分だけ分割時に多くとってくれるため一致しやすくなるが、とりすぎるとコーナー情報が被って歪みを消せなくなるため注意
    #bias = 150 

    #引数(画像、その画像の横幅、分割したい割合、分割したい個所から指定ピクセル多く作る(初期値は0))
    left_img_ref, right_img_ref = devide_image(img_ref,width,devide_rate,bias=0)

    #上底と下底の傾きと切片を求める
    upper_slope, lower_slope,upper_intercept,lower_intercept = get_slope(right_img_ref)
    
    #左側画像のコーナーをとる
    left_img_corner = get_leftside_points(left_img_ref)

    #傾きと切片から左側の画像の歪みをなくし、長方形になるようにする
    correct_img = homu_with_slope(left_img_ref,right_img_ref,upper_slope, lower_slope,upper_intercept,lower_intercept,left_img_corner)

    fusion_img = hconcat_resize_min([correct_img,right_img_ref])

    #left_fusion_img,right_fusion_img = devide_image(fusion_img,width,devide_rate,bias)

    #show2(fusion_img,right_fusion_img)
    #stitcing_fusion_img = stitcing(left_fusion_img,right_fusion_img)
    
    return fusion_img, upper_slope, lower_slope, upper_intercept, lower_intercept,DEVIDE_RATE,left_img_corner

#基板の部分のマスク画像を生成する関数
#supecularを使うこと前提
def trim_rect_as_binary(img):
    kernel_3 = np.ones((3,3),np.uint8)
    kernel_5 = np.ones((5,5),np.uint8)
    kernel_7 = np.ones((7,7),np.uint8)
    kernel_9 = np.ones((9,9),np.uint8)
    color = [255, 255, 255]
    
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    th = cv2.erode(th,kernel_3,iterations = 1)
    inv = cv2.bitwise_not(th)
    corner = get_corner_position(th)[0]
    #img_be_added_corner = cv2.drawContours(img, [corner], -1, (255,0,255), 3)
    umeume = cv2.fillPoly(th, [corner], color)
    erosion = cv2.erode(inv,kernel_9,iterations = 1)
    for i in range(2):
        erosion = cv2.erode(erosion,kernel_7,iterations = 1)
    inv = cv2.bitwise_not(erosion)
    dilation = cv2.dilate(inv,kernel_9,iterations = 1)
    for i in range(3):
        dilation = cv2.dilate(dilation,kernel_9,iterations = 1)
    
    return cv2.bitwise_not(cv2.bitwise_not(umeume)+cv2.bitwise_not(dilation))

#画像の位置合わせをする関数
def convert_image2rectangle(img):
    height,width=img.shape
    
    th = cv2.threshold(img, 80, 255, cv2.THRESH_OTSU)[1]
    erosion = cv2.erode(th,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations = 1)
    dilation = cv2.dilate(erosion,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations = 1)
    
    img_gray_corner = get_corner_position(dilation)[0]
    
    tgt_pts = np.float32([[0,height],[0,0],[width,0],[width,height]])
    #show1(correct_img)
    
    return convert_img_byUsingCorner(img_gray_corner, tgt_pts,img), img_gray_corner

#画像の位置合わせをする関数
def alignment(img):
    height,width=img.shape[:2]
    img_corner = get_leftside_points(img)

    #img_beadded_corner = cv2.drawContours(img, [img_corner], -1, (0,0,255), 3)
    
    min_h, max_h = get_right_side_corner(img)
    min_y, min_x = get_upper_left_corner(img)

    tgt_pts = np.float32([[img_corner[0,0], img_corner[0,1]],
                       [min_y, min_x],
                       [width,min_h],
                       [width,max_h]])

    #3090:4096=height:width(オリジナル画像)
    ref_pts = np.float32([[0,height],[0,0],[width,0],[width,height]])
    
    #下の引数をtgt_ptsにすると、右側の座標がすべて端っこになって、
    #img_gray_cornerにすると、単純に外接矩形になる
    correct_img = convert_img_byUsingCorner(tgt_pts, ref_pts,img)
    #show1(correct_img)
    
    return correct_img, tgt_pts

#画像最右側の閾値以上となる最初と最後の座標を取得する関数
def get_right_side_corner(img,threshold=150):
    min_h = 0
    max_h = img.shape[0]
    first = True
    for i in range(img.shape[0]):
        if img[i ,-1] > threshold:
            if first is True:
                min_h = i
                last_i = i
                first = False
            if i > last_i:
                max_h = i
    return min_h, max_h

#左上のコーナーを45度の角度で探索する関数
def get_upper_left_corner(img,threshold=150):
    min_x = 0
    min_y = 0
    minus = 20
    while True:
        if img[min_y ,min_x] > threshold:
            #ちょっと戻さないとめり込む
            return min_y-minus, min_x-minus
        if min_x==img.shape[1] or min_y==img.shape[0]:
            return
        min_x += 1
        min_y += 1

#ファイル内の拡張子を一括変更する関数
def change_extension(path, ref_extension="bmp", tgt_extension="png"):
    files = list(pathlib.Path(path).rglob("*." + ref_extension))
    for i,f in enumerate(files):
        print("{0}/{1}".format(i+1,len(files)))
        shutil.move(f, f.with_name(f.stem + "." + tgt_extension))

#OpenCV → Pillow
def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

#Pillow → OpenCV
def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

#ndarrayを回転させる関数
def rotate_img(img,angle):
    pil_img = cv2pil(img)
    kurukuru_pil_img = pil_img.rotate(angle)
    rotated_img = pil2cv(kurukuru_pil_img)
    return rotated_img

def shift_x(image, shift):
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,0] += shift # シフトするピクセル値
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (w, h))

def shift_y(image, shift):
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,1] += shift # シフトするピクセル値
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (w, h))

#画像をwidth×heightに分割して一番特徴点数の多かったトリミング箇所と座標を返す関数
def get_kp_in_rect(img, width, height):
    x, y = 0, 0
    good_x, good_y = 0, 0
    max_kp_num = 0
    
    # AKAZE作成
    akaze = cv2.AKAZE_create()
    
    if y+height >= img.shape[0] or x+width >= img.shape[1]:      
        return print("Error : widthまたはheightが大きすぎます。")
    while y+height < img.shape[0]:
        while x+width < img.shape[1]:      
            cropped = crop_image(img, x, y, width, height)
            kp = akaze.detectAndCompute(cropped, None)[0]
            if len(kp)>max_kp_num:
                max_kp_num = len(kp)
                good_x = x
                good_y = y
            x += width
        y += height
        x = 0
    cropped = crop_image(img, good_x, good_y, width, height)   
    return cropped, good_x, good_y

#切り出し関数
def crop_image(img, x, y, width, height):
    cropped = img[y:height+y, x:width+x]
    return cropped

#パラボラフィッティングによるサブピクセル推定を行う関数
def get_subPixel_using_parabola_fiting(match_result):
    #最も一致率の高い座標を取得(minMaxLocの返り値を渡すのもありだったよね)
    match_index = np.unravel_index(np.argmax(match_result), match_result.shape)
    
    #最も一致率の高い座標の前後(x座標)のピクセルの一致率を取得
    R_back, R_center, R_front = match_result[match_index[0], match_index[1]-1] , \
                                match_result[match_index[0], match_index[1]] , \
                                match_result[match_index[0], match_index[1]+1]
    #サブピクセルの計算
    sub_pixel = (R_back - R_front) /  (2*R_back - 4*R_center + 2*R_front)
    #print(R_back, R_center, R_front)
    return sub_pixel

#テンプレートマッチ箇所を四角で囲む関数
def surround_matchpoint_rect(ref_img, tgt_img, save_dir):
    #ターゲット画像に対してトリミング画像を用いて、テンプレートマッチング
    match_result = cv2.matchTemplate(ref_img, tgt_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)

    #検出領域を四角で囲んで保存
    w, h = tgt_img.shape[::-1]
    result = ref_img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(result,top_left, bottom_right, (0,0,255), 3)
    cv2.imwrite(save_dir, result)