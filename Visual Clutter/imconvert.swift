//
//  imconvert.swift
//  Visual Clutter
//
//  Created by Monash Assistive Tech Team on 24/10/2024.
//

import Foundation
import UIKit

import UIKit
import CoreVideo

extension UIImage {
    // Initialize UIImage from CVPixelBuffer
    convenience init?(pixelBuffer: CVPixelBuffer) {
        // Lock the base address of the pixel buffer for reading
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags.readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags.readOnly) }

        let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        if let context = CGContext(data: baseAddress,
                                    width: width,
                                    height: height,
                                    bitsPerComponent: 8,
                                    bytesPerRow: bytesPerRow,
                                    space: colorSpace,
                                    bitmapInfo: bitmapInfo.rawValue),
           let cgImage = context.makeImage() {
            self.init(cgImage: cgImage)
        } else {
            return nil
        }
    }
}
