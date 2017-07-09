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
import SpeechToTextV1

class HomeController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate, AVAudioPlayerDelegate, AVAudioRecorderDelegate {
    
    // asdf
    var speechToText: SpeechToText!
    var speechToTextSession: SpeechToTextSession!
    var isStreaming = false
    
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
        streamMicrophoneBasic()
    }
    
    /*
    This function demonstrates how to use the basic
    `SpeechToText` class to transcribe microphone audio.
    */
    public func streamMicrophoneBasic() {
        if !isStreaming {
            
            // update state
            noise.setTitle("Stop Microphone", for: .normal)
            isStreaming = true
            
            // define recognition settings
            var settings = RecognitionSettings(contentType: .opus)
            settings.continuous = true
            settings.interimResults = true
            
            // define error function
            let failure = { (error: Error) in print(error) }
            
            // start recognizing microphone audio
            speechToText.recognizeMicrophone(settings: settings, failure: failure) {
                results in
                self.textView.text = results.bestTranscript
            }
            
        } else {
            
            // update state
            noise.setTitle("Start Microphone", for: .normal)
            isStreaming = false
            
            // stop recognizing microphone audio
            speechToText.stopRecognizeMicrophone()
        }
    }
    
    /**
     This function demonstrates how to use the more advanced
     `SpeechToTextSession` class to transcribe microphone audio.
     */
    public func streamMicrophoneAdvanced() {
        if !isStreaming {
            
            // update state
            noise.setTitle("Stop Microphone", for: .normal)
            isStreaming = true
            
            // define callbacks
            speechToTextSession.onConnect = { print("connected") }
            speechToTextSession.onDisconnect = { print("disconnected") }
            speechToTextSession.onError = { error in print(error) }
            speechToTextSession.onPowerData = { decibels in print(decibels) }
            speechToTextSession.onMicrophoneData = { data in print("received data") }
            speechToTextSession.onResults = { results in self.textView.text = results.bestTranscript }
            
            // define recognition settings
            var settings = RecognitionSettings(contentType: .opus)
            settings.continuous = true
            settings.interimResults = true
            
            // start recognizing microphone audio
            speechToTextSession.connect()
            speechToTextSession.startRequest(settings: settings)
            speechToTextSession.startMicrophone()
            
        } else {
            
            // update state
            noise.setTitle("Start Microphone", for: .normal)
            isStreaming = false
            
            // stop recognizing microphone audio
            speechToTextSession.stopMicrophone()
            speechToTextSession.stopRequest()
            speechToTextSession.disconnect()
        }
    }
    
    func timerCalled(timer: Timer) {
        capturePhoto()
        print("HEYYYYYYYYYYYYYY")
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        speechToText = SpeechToText(
            username: Credentials.SpeechToTextUsername,
            password: Credentials.SpeechToTextPassword
        )
        speechToTextSession = SpeechToTextSession(
            username: Credentials.SpeechToTextUsername,
            password: Credentials.SpeechToTextPassword
        )
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

