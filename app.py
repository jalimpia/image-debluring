from flask import Flask,render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import cv2 as cv
import numpy as np
from algorithm import surf, brisk, briskd, proposed, fft, wiener
import os
import time
import hashlib
from datetime import datetime

from keras import backend as K

from deblurgan.model import generator_model
from deblurgan.utils import load_image, deprocess_image, preprocess_image
from PIL import Image



INPUT_FOLDER = os.path.join('static', 'input_images')
OUTPUT_FOLDER = os.path.join('static', 'output_images')
weight_path=os.path.join('static/weight', 'generator.h5')


app = Flask(__name__)
app.secret_key = "BRISD_SIMULATION"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/extract1', methods=['POST'])
def extract1():
     now = datetime.now() # current date and time
     date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
     if request.method == 'POST':
        reference = request.files['reference_img']
        reference_image = secure_filename(reference.filename)
        path = os.path.join(INPUT_FOLDER, reference_image)
        reference.save(path)
        result, keypoints, time = briskd.compute_keypoints(reference_image)

        digest = hashlib.md5()
        digest.update(date_time.encode('utf-8'))

        output_path = os.path.join(OUTPUT_FOLDER, digest.hexdigest()+reference_image)
        cv.imwrite(output_path, result)
        result_table = f'''
        <table class="table">
            <tr>
                <th>Keypoints</th>
                <th>Time</th>
            </tr>
            <tr>
                <td>{keypoints}</td>
                <td>{time} ms</td>
            </tr>
            <tr>
                <td colspan="2">
                    <img src="{output_path}" class="img-fluid">
                </td>
            </tr>
        </table>
        '''
        #Set Global Variable
        session['reference_image'] = path
        return str(result_table)

@app.route('/extract2', methods=['POST'])
def extract2():
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    if request.method == 'POST':
        sample = request.files['sample_img']
        threshold = request.form['threshold']
        sample_image = secure_filename(sample.filename)
        path = os.path.join(INPUT_FOLDER, sample_image)
        sample.save(path)
        #Check if blur or not based on threshold
        msk, val, blurry = fft.detect_blur(sample_image,threshold)

        digest = hashlib.md5()
        digest.update(date_time.encode('utf-8'))

        output_path = os.path.join(OUTPUT_FOLDER, digest.hexdigest()+sample_image)
        cv.imwrite(output_path, msk)
        result_table = f'''
        <table class="table">
            <tr>
                <th>Score</th>
                <th>Is Blurry</th>
            </tr>
            <tr>
                <td>{val}</td>
                <td>{blurry} </td>
            </tr>
            <tr>
                <td colspan="2">
                    <img src="{output_path}" class="img-fluid">
                </td>
            </tr>
        </table>
        '''
        #Set Global Variable
        session['sample_image'] = path
        return str(result_table)



@app.route('/deblurme', methods=['POST'])
def deblurme():
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    if request.method == 'POST':
        g = generator_model()
        g.load_weights(weight_path)
        sample = request.files['sample_img']
        sample_image = secure_filename(sample.filename)
        path = os.path.join(INPUT_FOLDER, sample_image)
        sample.save(path)

        #START TIME
        start_time = time.time()

        image = np.array([preprocess_image(load_image(os.path.join(INPUT_FOLDER, sample_image)))])
        generated_images = g.predict(x=image)
        #x_test = image
        generated = np.array([deprocess_image(img) for img in generated_images])
        x_test = deprocess_image(image)

        #END TIME
        end_time = time.time()
        elapsed_time = end_time - start_time

        #Clear TF Session
        K.clear_session()

        img = generated[0, :, :, :]
        im = Image.fromarray(img.astype(np.uint8))

        sample_size = Image.open(path)
        im = im.resize(sample_size.size)

        #For Comparison
        #x = x_test[i, :, :, :]
        #img = np.concatenate((x, img), axis=1)

        digest = hashlib.md5()
        digest.update(date_time.encode('utf-8'))
        output_path = os.path.join(OUTPUT_FOLDER, digest.hexdigest()+sample_image)

        im.save(output_path)

        result_table = f'''
        <table class="table" style="width:100%">
            <tr>
                <th class="text-center">Processing Time</th>
                <th class="text-center">Path</th>
            </tr>
            <tr>
                <td class="text-center">
                    {elapsed_time} sec
                </td>
                <td class="text-center">
                    <a href="{output_path}" target="blank" download>Download</a>
                </td>
            </tr>
            <tr>
                <td class="text-center">
                    <img width="350" height="350" src="{path}"  /><br>
                    <p class="badge">Orignal Image</p>
                </td>
                <td class="text-center">
                    <img width="350" height="350" src="{output_path}"  /><br>
                    <p class="badge">Refined Image</p>
                </td>
            </tr>
        </table>
        '''
        #Set Global Variable
        session['refined_image'] = output_path
        return str(result_table)

result_count = 0
@app.route('/compare', methods=['POST'])
def compare():
        if request.method == 'POST':
            global result_count
            result_count+=1
            # sample_image = cv.imread(session['sample_image'], cv2.IMREAD_UNCHANGED)
            # refined_image = cv.imread(session['refined_image'], cv2.IMREAD_UNCHANGED)
            # resized = cv.resize(refined_image, sample_image.shape, interpolation = cv2.INTER_AREA)

            result1, matches1, time1 = briskd.compute_matches(session['reference_image'],session['refined_image'])
            result2, matches2, time2 = briskd.compute_matches(session['reference_image'],session['sample_image'])
            result_table = f'''
            <table class="table" style="width:100%">
                <tr>
                    <th class="text-center">Algorithm</th>
                    <th class="text-center">Processing Time</th>
                    <th class="text-center">Matches</th>
                </tr>
                <tr>
                    <td class="text-center">
                        BRISK_D2
                    </td>
                    <td class="text-center" id="BRISK_D2_TIME">
                        {time1} ms
                    </td>
                    <td class="text-center" id="BRISK_D2_MATCHES">
                        {matches1}
                    </td>
                </tr>
                <tr>
                    <td colspan="3" class="text-center">
                        <img id="BRISK_D2_IMG" src="{result1}" class="img-fluid"/><br>
                        <p class="badge">Keypoint Matching for BRISK_D2</p>
                    </td>
                </tr>
                <tr>
                    <td class="text-center">
                        BRISK_D
                    </td>
                    <td class="text-center" id="BRISK_D_TIME">
                        {time2} ms
                    </td>
                    <td class="text-center" id="BRISK_D_MATCHES">
                        {matches2}
                    </td>
                </tr>
                <tr>
                    <td colspan="3" class="text-center">
                        <img id="BRISK_D_IMG" src="{result2}" class="img-fluid"/><br>
                        <p class="badge">Keypoint Matching for BRISK_D</p>
                    </td>
                </tr>
            </table>
            <script>
              $('#row_results').append(`
                <tr>
                  <td>{result_count}</td>
                  <td>{round(time2,2)} ms</td>
                  <td>{matches2}</td>
                  <td>{round(time1,2)} ms</td>
                  <td>{matches1}</td>
                  <td>{round( abs(int(matches2) - int(matches1)) / int(matches1) * 100, 2)} %</td>
                </tr>
              `);
              </script>
            '''
            return str(result_table)


@app.route("/matching", methods=['POST'])
def matching():

    algo = request.form['algo']
    image1 = request.files['image1']
    image2 = request.files['image2']
    img_name1 = secure_filename(image1.filename)
    img_name2 = secure_filename(image2.filename)

    path1 = os.path.join(INPUT_FOLDER, img_name1)
    path2 = os.path.join(INPUT_FOLDER, img_name2)
    image1.save(path1)
    image2.save(path2)

    if algo=='SURF':
        result, matches, time = surf.compute_matches(img_name1,img_name2)
    elif algo=='BRISK':
        result, matches, time = brisk.compute_matches(img_name1,img_name2)
    elif algo=='BRISKD':
        result, matches, time = briskd.compute_matches(img_name1,img_name2)
    elif algo=='PROPOSED':
        result, matches, time = proposed.compute_matches(img_name1,img_name2)
    elif algo=='ALL':
        result1, matches1, time1 = surf.compute_matches(img_name1,img_name2)
        result2, matches2, time2 = brisk.compute_matches(img_name1,img_name2)
        result3, matches3, time3 = briskd.compute_matches(img_name1,img_name2)
        result4, matches4, time4 = proposed.compute_matches(img_name1,img_name2)
        return f'''
        <h2>Table Comparison</h2>
        <p>The comparison about correct matching number of feature points and time consumption (/ms) by each algorithm.</p>
        <label>Results: </label>
        <table border=1 style="width:380px">
          <tr>
            <th>Algorithm</th>
            <th>Matches</th>
            <th>Time</th>
          </tr>
          <tr>
            <td>SURF</td>
            <td>{matches1} keypoints</td>
            <td>{time1} ms</td>
          </tr>
          <tr>
            <td>BRISK</td>
            <td>{matches2} keypoints</td>
            <td>{time2} ms</td>
          </tr>
          <tr>
            <td>BRISKD</td>
            <td>{matches3} keypoints</td>
            <td>{time3} ms</td>
          </tr>
          <tr>
            <td>PROPOSED</td>
            <td>{matches4} keypoints</td>
            <td>{time4} ms</td>
          </tr>
        </table>
        <a href="/">Back</a>
        '''

    output_name= algo+'_matching_output_'+str(matches)+'.jpg'
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    cv.imwrite(output_path, result)

    return render_template('index.html', algo=algo.upper(), matches=matches, time=time, result=output_path)

@app.route("/detecting", methods=['POST'])
def detecting():
    algo = request.form['algo']
    image1 = request.files['image1']
    img_name1 = secure_filename(image1.filename)
    path1 = os.path.join(INPUT_FOLDER, img_name1)
    image1.save(path1)

    if algo=='SURF':
        result, quantity, time = surf.compute_keypoints(img_name1)
    elif algo=='BRISK':
        result, quantity, time = brisk.compute_keypoints(img_name1)
    elif algo=='BRISKD':
        result, quantity, time = briskd.compute_keypoints(img_name1)
    elif algo=='PROPOSED':
        result, quantity, time = briskd.compute_keypoints(img_name1)
    elif algo=='ALL':
        result1, quantity1, time1 = surf.compute_keypoints(img_name1)
        result2, quantity2, time2 = brisk.compute_keypoints(img_name1)
        result3, quantity3, time3 = briskd.compute_keypoints(img_name1)
        result4, quantity4, time4 = briskd.compute_keypoints(img_name1)
        return f'''
        <h2>Table Comparison</h2>
        <p>The comparison about feature points extracted by each algorithm and their time consumption.</p>
        <label>Results: </label>
        <table border=1 style="width:380px">
          <tr>
            <th>Algorithm</th>
            <th>Quantity</th>
            <th>Time</th>
          </tr>
          <tr>
            <td>SURF</td>
            <td>{quantity1} keypoints</td>
            <td>{time1} ms</td>
          </tr>
          <tr>
            <td>BRISK</td>
            <td>{quantity2} keypoints</td>
            <td>{time2} ms</td>
          </tr>
          <tr>
            <td>BRISKD</td>
            <td>{quantity3} keypoints</td>
            <td>{time3} ms</td>
          </tr>
          <tr>
            <td>PROPOSED</td>
            <td>{quantity4} keypoints</td>
            <td>{time4} ms</td>
          </tr>
        </table>
        <a href="/">Back</a>
        '''

    output_name= algo+'_detecting_output_'+str(quantity)+'.jpg'
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    cv.imwrite(output_path, result)

    return render_template('index.html', algo=algo.upper(), quantity=quantity, time=time, result=output_path)

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)


# image, matches, time = surf.compute_matches("1.png","2.png")
# cv.imwrite('./output_images/sample.png', image)
# print(matches)
# print(time)

# image, quantity, time = surf.compute_keypoints("python1.jpg")
# cv.imshow('image',image)
# cv.waitKey(0)
