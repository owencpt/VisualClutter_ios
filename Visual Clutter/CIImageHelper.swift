//
//  CIImageHelper.swift
//  Visual Clutter
//
//  Created by Monash Assistive Tech Team on 26/1/2025.
//

import Foundation
import CoreImage
import CoreVideo

// MARK: - CIImage to CVPixelBuffer Conversion
extension CIImage {
    func toPixelBuffer() -> CVPixelBuffer? {
        // Define the attributes for the pixel buffer
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(extent.width),  // Width of the image
            Int(extent.height), // Height of the image
            kCVPixelFormatType_32BGRA, // Pixel format (32-bit BGRA)
            attrs,
            &pixelBuffer
        )

        // Check if the pixel buffer was created successfully
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        // Render the CIImage into the pixel buffer
        let context = CIContext()
        context.render(self, to: buffer)

        return buffer
    }
}

// MARK: - Image Resizing
func resizeImage(_ image: CIImage, to size: CGSize) -> CIImage {
    let scaleX = size.width / image.extent.width
    let scaleY = size.height / image.extent.height
    return image.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
}
