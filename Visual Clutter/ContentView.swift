//
//  ContentView.swift
//  Visual Clutter
//
//  Created by Monash Assistive Tech Team on 2/10/2024.
//

import SwiftUI
import CoreMedia
import AVFoundation
import Vision


// UIViewRepresentable to bridge UIKit's AVCaptureVideoPreviewLayer to SwiftUI
struct CameraView: UIViewRepresentable {
    let videoCapture: VideoCapture
    
    func makeCoordinator() -> Coordinator {
        Coordinator(videoCapture: videoCapture)
    }

    class Coordinator {
        var videoCapture: VideoCapture
        
        init(videoCapture: VideoCapture) {
            self.videoCapture = videoCapture
        }
    }
    
    // Creates the UIView to display the camera preview
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        
        // Set up the camera session
        videoCapture.setUp { success in
            if success {
                DispatchQueue.main.async {
                    if let previewLayer = self.videoCapture.previewLayer {
                        previewLayer.frame = view.bounds
                        view.layer.addSublayer(previewLayer)
                        print("Preview layer added")
                    } else {
                        print("No preview layer available.")
                    }
                }
            } else {
                print("Failed to set up camera.")
            }
        }
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
            // Update the preview layer's frame
            videoCapture.previewLayer?.frame = uiView.bounds
        
    
    }

}

class BoundingBoxManager: ObservableObject {
    @Published var boundingBoxes: [CGRect] = []
}

// SwiftUI view containing the camera preview
struct ContentView: View {
    
    @ObservedObject var videoCapture = VideoCapture()
    let items = ["cup", "bottle", "knife", "spoon", "laptop", "scissors"]
    
    @State private var selectedItem: String? = nil
    @State private var isMenuOpen = false
    @State private var modelStatus = false
    
//    @State private var scanTimeRemaining = 10
//    @State private var timer: Timer?
//    @State private var showTimeoutMessage = false



    
    
    var body: some View {
        ZStack{
            if modelStatus{
                CameraView(videoCapture: videoCapture)
                    .edgesIgnoringSafeArea(.all)
                
                
                if videoCapture.rect != .zero {
                    DarkenedOverlayView(boundingRect: videoCapture.rect)
                }
                
                // Add additional UI elements only when processedImage is nil
                if videoCapture.processedImage == nil {
                    VStack {
                        if videoCapture.rect == .zero {
                            // If rect is .zero, it means no object was detected.
                            Text("No \(videoCapture.selected) detected. Please move the camera slowly to scan the area.")
                                .foregroundColor(.white)
                                .font(.headline)
                                .multilineTextAlignment(.center)
                                .padding()
                                .background(Color.black.opacity(0.9))
                                .cornerRadius(10)
                                .padding(.top, 50)
                                .padding(.horizontal, 20)  // Add padding to the left and right for spacing
                                .frame(maxWidth: 350)      // Set a max width to avoid text box stretching too wide
                                .lineLimit(2)              // Allow up to 2 lines
                                .minimumScaleFactor(0.5)   // Scale down the text if it exceeds the space

                        } else {
                            // If rect is not .zero, it means an object was detected.
                            Text("You have found the \(videoCapture.selected)!")
                                .foregroundColor(.white)
                                .font(.headline)
                                .multilineTextAlignment(.center)
                                .padding()
                                .background(Color.black.opacity(0.9))
                                .cornerRadius(10)
                                .padding(.top, 50)
                        }

                        Spacer() // Push the text to the top
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity) // Ensure the VStack takes up the full screen
                }

                        

            }else{
                
                Color.white
                    .ignoresSafeArea(.all)
                
                Text("CVI Clutter Tool Kit")
                    .font(.system(size: 30, weight: .bold, design: .rounded))
                    .foregroundColor(.black) // Neutral text color
                    .padding(15) // Add padding inside the border
                    .background(
                        RoundedRectangle(cornerRadius: 20) // Rounded border
                            .fill(Color.white) // White background
                            .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2) // Subtle shadow
                            .overlay(
                                RoundedRectangle(cornerRadius: 20)
                                    .stroke(Color.gray.opacity(0.3), lineWidth: 1) // Subtle border
                            )
                    )
                
                VStack {
                    HStack {
                        Text("MATT")
                            .font(.system(size: 12, weight: .bold, design: .rounded))
                            .foregroundColor(.black) // Neutral text color
                            .padding(15) // Add padding inside the border
                            .background(
                                RoundedRectangle(cornerRadius: 20) // Rounded border
                                    .fill(Color.white) // White background
                                    .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2) // Subtle shadow
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 20)
                                            .stroke(Color.gray.opacity(0.3), lineWidth: 1) // Subtle border
                                    )
                            )
                        
                            .padding(.horizontal,20)

                    
                        Spacer() // Push the text to the left
                    }
                    Spacer() // Push the HStack to the top
                }
                
            }
            
            
            if modelStatus{
                Rectangle()
                    .stroke(Color.red, lineWidth: 2)
                    .frame(width: videoCapture.rect.width,
                           height: videoCapture.rect.height)
                    .position(x: videoCapture.rect.midX ,
                              y: (videoCapture.rect.midY)) // Flip Y-axis for Vision bounding box
            }
            
        
            VStack(spacing: 20) {
                
                Spacer()
                
                Button(action: {
                    if modelStatus {
                        self.videoCapture.stop()
                    }
                    else{
                        
                        self.videoCapture.start()
                    }
                    modelStatus.toggle()
                }, label: {
                    Text(modelStatus ? "Stop Scan" : "Start Scan")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .frame(width: 200, height: 50)
                        .background(Color(hex: "#355070"))
                        .cornerRadius(25)
                })
                
                Button(action: {
                    isMenuOpen.toggle() // Toggle the menu state
                }, label: {
                    HStack {
                        Text(videoCapture.selected.capitalized)
                            .font(.headline)
                            .foregroundColor(.white)
                        
                        // Add an icon (e.g., a chevron down) next to the text
                        Image(systemName: isMenuOpen ? "chevron.up" : "chevron.down") // Change icon based on menu state
                            .foregroundColor(.white)
                            .font(.system(size:12 ,weight: .black))
                    }
                    .padding()
                    .frame(width: 200, height: 50)
                    .background(Color(hex: "#818589"))
                    .cornerRadius(25)
                })
                
            
            }
            

            
            if isMenuOpen{
                                
                VStack {
                    
//                    Spacer().frame(height: 50)
                    
//                    Text("Select Item For Detection")
//                        .font(.headline)
                        

                    ScrollView {
                        
                        LazyVGrid(
                            columns: [
                                GridItem(.flexible(), spacing: 16), // First column
                                GridItem(.flexible(), spacing: 16)  // Second column
                            ],
                            spacing: 16 // Vertical spacing between rows
                        ) {
                            ForEach(items, id: \.self) { item in
                                Button(action: {
                                    chooseItem(item: item)
                                    isMenuOpen = false
                                }) {
                                    Text(item.capitalized)
                                        .font(.body)
                                        .fontWeight(.semibold)
                                        .padding()
                                        .frame(maxWidth: .infinity)
                                        .background(
                                            RoundedRectangle(cornerRadius: 20) // Rounded border
                                                .fill(Color.white) // White background
                                                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2) // Subtle shadow
                                                .overlay(
                                                    RoundedRectangle(cornerRadius: 20)
                                                        .stroke(Color.gray.opacity(0.3), lineWidth: 1) // Subtle border
                                                )
                                        )
                                        .cornerRadius(8)
                                        .foregroundColor(Color.black)
                                }
                                .padding(.horizontal, 16) // Adjust horizontal padding inside each grid cell
                            }
                        }
                        .padding(16)
                    }
                    .background(Color.white)
                    .cornerRadius(12)
                    .shadow(radius: 5)
                    .frame(maxHeight: 225)
                    .padding(.horizontal,10)
                }
            }
        }
    }

    // Helper function to request camera permission
    func requestCameraPermission(completion: @escaping () -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            completion()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    DispatchQueue.main.async {
                        completion()
                    }
                } else {
                    print("Camera access denied.")
                }
            }
        case .denied, .restricted:
            print("Camera access restricted or denied.")
        @unknown default:
            print("Unknown camera authorization status.")
        }
    }

    // Loads the model (activated by the start button)
    func loadModel(){
        requestCameraPermission {
            self.videoCapture.start()
        }
    }
    
    // alters the current item and sends it to videocapture for processing. (Used by each button on the object menu)
    func chooseItem(item:String){
        videoCapture.selected = item
                
    }
}


struct DarkenedOverlayView: View {
    let boundingRect: CGRect
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Top overlay
                Rectangle()
                    .fill(Color.black.opacity(0.75))
                    .frame(width: geometry.size.width,
                           height: boundingRect.minY+80) // Keeps the height the same
                    .position(x: geometry.size.width / 2,
                              y: boundingRect.minY / 2+6) // This will position it above the bounding box
                
                // Bottom overlay
                Rectangle()
                    .fill(Color.black.opacity(0.75))
                    .frame(width: geometry.size.width,
                           height: geometry.size.height - boundingRect.maxY)
                    .position(x: geometry.size.width / 2,
                              y: geometry.size.height - (geometry.size.height - boundingRect.maxY) / 2+48)
                
                // Left overlay
                Rectangle()
                    .fill(Color.black.opacity(0.75))
                    .frame(width: boundingRect.minX,
                           height: boundingRect.height+2)
                    .position(x: boundingRect.minX / 2,
                              y: boundingRect.midY + 47)
                
                // Right overlay
                Rectangle()
                    .fill(Color.black.opacity(0.75))
                    .frame(width: geometry.size.width - boundingRect.maxX,
                           height: boundingRect.height+2)
                    .position(x: geometry.size.width - (geometry.size.width - boundingRect.maxX) / 2,
                              y: boundingRect.midY+47)
            }
            .allowsHitTesting(false) // Ensure this doesn't block interactions
        }
        .edgesIgnoringSafeArea(.all)
    }
}







