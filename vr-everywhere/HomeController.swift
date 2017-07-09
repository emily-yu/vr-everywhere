//
//  HomeController.swift
//  vr-everywhere
//
//  Created by Emily on 7/8/17.
//  Copyright © 2017 Emily. All rights reserved.
//

import UIKit
import Foundation
import AVFoundation
import Photos
import Alamofire

class HomeController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate, AVAudioPlayerDelegate, AVAudioRecorderDelegate {
    
    var session: AVCaptureSession!
    var input: AVCaptureDeviceInput!
    var output: AVCaptureStillImageOutput!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var base64String: String?
    var ngrok = "https://4adacd3b.ngrok.io/send?input=";
    
    var audioPlayer: AVAudioPlayer?
    var audioRecorder: AVAudioRecorder?
    
    @IBOutlet var textView: UITextView!
    
    @IBAction func trigger(_ sender: Any) {
        self.setupSession()
//        var cameraTimer = Timer.scheduledTimer(timeInterval: 20, target: self, selector: #selector(HomeController.timerCalled), userInfo: nil, repeats: false)
        capturePhoto()
//        let when = DispatchTime.now() + 4 // change 2 to number of seconds to last for
//        DispatchQueue.main.asyncAfter(deadline: when) {
//            cameraTimer.invalidate()
//        }
    }
    
    // record
    @IBOutlet var noise: UIButton!
    @IBAction func noise(_ sender: Any) {
        print("noise")
        if audioRecorder?.isRecording == false {
            play.isEnabled = false
            stop.isEnabled = true
            audioRecorder?.record()
        }
    }
    
    @IBOutlet var stop: UIButton!
    @IBAction func stop(_ sender: Any) {
        print("stop")
        stop.isEnabled = false
        play.isEnabled = true
        noise.isEnabled = true
        
        if audioRecorder?.isRecording == true {
            audioRecorder?.stop()
            
            // transcribe to text here
            
            
        } else {
            audioPlayer?.stop()
        }
    }
    
    @IBOutlet var play: UIButton!
    @IBAction func play(_ sender: Any) {
        print("play")
        if audioRecorder?.isRecording == false {
            stop.isEnabled = true
            noise.isEnabled = false
            
            do {
                try audioPlayer = AVAudioPlayer(contentsOf:
                    (audioRecorder?.url)!)
                audioPlayer!.delegate = self
                audioPlayer!.prepareToPlay()
                audioPlayer!.play()
            } catch let error as NSError {
                print("audioPlayer error: \(error.localizedDescription)")
            }
        }
    }
    
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        noise.isEnabled = true
        stop.isEnabled = false
    }
    
    func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        print("Audio Play Decode Error")
    }
    
    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
    }
    
    func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: Error?) {
        print("Audio Record Encode Error")
    }
    
    func timerCalled(timer: Timer) {
        capturePhoto()
        print("HEYYYYYYYYYYYYYY")
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        play.isEnabled = false
        stop.isEnabled = false
        
        let fileMgr = FileManager.default
        
        let dirPaths = fileMgr.urls(for: .documentDirectory,
                                    in: .userDomainMask)
        
        let soundFileURL = dirPaths[0].appendingPathComponent("sound.caf")
        
        let recordSettings =
            [AVEncoderAudioQualityKey: AVAudioQuality.min.rawValue,
             AVEncoderBitRateKey: 16,
             AVNumberOfChannelsKey: 2,
             AVSampleRateKey: 44100.0] as [String : Any]
        
        let audioSession = AVAudioSession.sharedInstance()
        
        do {
            try audioSession.setCategory(
                AVAudioSessionCategoryPlayAndRecord)
        } catch let error as NSError {
            print("audioSession error: \(error.localizedDescription)")
        }
        
        do {
            try audioRecorder = AVAudioRecorder(url: soundFileURL,
                                                settings: recordSettings as [String : AnyObject])
            audioRecorder?.prepareToRecord()
        } catch let error as NSError {
            print("audioSession error: \(error.localizedDescription)")
        }
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    func setupSession() {
        print("haaaeaea")
        print(ngrok)
        session = AVCaptureSession()
        session.sessionPreset = AVCaptureSessionPresetPhoto
        
        let camera = AVCaptureDevice
            .defaultDevice(withMediaType: AVMediaTypeVideo)
        
        do { input = try AVCaptureDeviceInput(device: camera) } catch { return }
        
        output = AVCaptureStillImageOutput()
        output.outputSettings = [ AVVideoCodecKey: AVVideoCodecJPEG ]
        
        guard session.canAddInput(input)
            && session.canAddOutput(output) else { return }
        
        session.addInput(input)
        session.addOutput(output)
        
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        
        previewLayer!.videoGravity = AVLayerVideoGravityResizeAspect
        previewLayer!.connection?.videoOrientation = .portrait
        
        view.layer.addSublayer(previewLayer!)
        
        session.startRunning()
    }
    
    func capturePhoto() {
        guard let connection = output.connection(withMediaType: AVMediaTypeVideo) else { return }
        connection.videoOrientation = .portrait
        
        output.captureStillImageAsynchronously(from: connection) { (sampleBuffer, error) in
            guard sampleBuffer != nil && error == nil else { return }
            
            let imageData = AVCaptureStillImageOutput.jpegStillImageNSDataRepresentation(sampleBuffer)
            guard let image = UIImage(data: imageData!) else { return }
//            
            let imageJPG: Data! = UIImageJPEGRepresentation(image, 0.1)
//            let base64String = (imageJPG as NSData).base64EncodedString(options: NSData.Base64EncodingOptions(rawValue: 0))
            Alamofire.request("\(self.ngrok)\((imageJPG as NSData).base64EncodedString(options: NSData.Base64EncodingOptions(rawValue: 0)))")
            print("\(self.ngrok)\((imageJPG as NSData).base64EncodedString(options: NSData.Base64EncodingOptions(rawValue: 0)))")
//            self.ngrok += "\(base64String)%" // splice it at %
            
//            PHPhotoLibrary.shared().performChanges({
//                PHAssetChangeRequest.creationRequestForAsset(from: image)
//            }, completionHandler: { success, error in
//                if success {
//                    // Saved successfully!
//                }
//                else if let error = error {
//                    // Save photo failed with error
//                }
//                else {
//                    // Save photo failed with no error
//                }
//            })
        }
    }

    
}

