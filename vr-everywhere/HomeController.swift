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

class HomeController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    @IBAction func trigger(_ sender: Any) {
        self.setupSession()
        var cameraTimer = Timer.scheduledTimer(timeInterval: 0.1, target: self, selector: #selector(HomeController.timerCalled), userInfo: nil, repeats: true)
        let when = DispatchTime.now() + 2 // change 2 to number of seconds to last for
        DispatchQueue.main.asyncAfter(deadline: when) {
            cameraTimer.invalidate()
            Alamofire.request(self.ngrok) // send shit seperated by %
            console.log(self.ngrok)
        }
    }
    
    func timerCalled(timer: Timer) {
        capturePhoto()
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    var session: AVCaptureSession!
    var input: AVCaptureDeviceInput!
    var output: AVCaptureStillImageOutput!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var count = 0;
    var ngrok = "http://41e888fa.ngrok.io/send?input=";
    
    func setupSession() {
        print("haaaeaea")
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
        count += 1 // add one to te number of sihtters to
        print("same")
        guard let connection = output.connection(withMediaType: AVMediaTypeVideo) else { return }
        connection.videoOrientation = .portrait
        
        output.captureStillImageAsynchronously(from: connection) { (sampleBuffer, error) in
            guard sampleBuffer != nil && error == nil else { return }
            
            let imageData = AVCaptureStillImageOutput.jpegStillImageNSDataRepresentation(sampleBuffer)
            guard let image = UIImage(data: imageData!) else { return }
            
            let imageJPG: Data! = UIImageJPEGRepresentation(image, 0.1)
            let base64String = (imageJPG as NSData).base64EncodedString(options: NSData.Base64EncodingOptions(rawValue: 0))
            self.ngrok += base64String + "%" // splice it at %
        }
    }

    
}

