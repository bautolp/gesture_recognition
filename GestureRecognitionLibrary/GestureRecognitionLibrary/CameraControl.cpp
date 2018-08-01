/* This Gesture recognition software is provided by Lyndon Bauto (github.com/bautolp) "as is" and "with all faults." I
 * make no representations or warranties of any kind concerning the safety, suitability, lack of viruses,
 * inaccuracies, typographical errors, or other harmful components of this Gesture recognition software. There are
 * inherent dangers in the use of any software, and you are solely responsible for determining whether this  Gesture
 * recognition software is compatible with your equipment and other software installed on your equipment.You are also
 * solely responsible for the protection of your equipment and backup of your data, and I will not be liable
 * for any damages you may suffer in connection with using, modifying, or distributing this Gesture recognition
 * software. */

#include "CameraControl.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <time.h>
#include <urlmon.h>
#include <vector>

using namespace std;
using namespace Arena;
using namespace GenApi;
using namespace cv;
using namespace cuda;
using namespace chrono;

#define ENUM_TIMEOUT_S 3.0
#define IMAGE_TIMEOUT 100
#define RETRY_CNT 50

CameraControl::CameraControl()
{
	m_device = NULL;
	m_node_map = NULL;
	m_system = NULL;
	cv_image = NULL;
}

CameraControl::~CameraControl()
{
}

cv::cuda::GpuMat* CameraControl::GetCompleteMat()
{
	if (!m_device->IsConnected())
	{
		throw exception("Lost connection to device.");
	}

	IImage* image = NULL;

	image = m_device->GetImage(200);

	unsigned int retry_count = 0;
	bool image_incomplete = image->IsIncomplete();
	while (image_incomplete)
	{
		// Free and grab again
		m_device->RequeueBuffer(image);
		image = m_device->GetImage(200);
		if (++retry_count > RETRY_CNT)
		{
			throw exception(string("Cannot get valid image data after " + to_string(RETRY_CNT) + " retries.").c_str());
		}
		image_incomplete = image->IsIncomplete();
	}

	if (image->GetWidth() == 0 || image->GetHeight() == 0 || image->GetPixelFormat() == 0)
	{
		throw exception("width, height, and pixel format must be nonzero.");
	}

	// Kind of redundant?
	if (image->GetPixelFormat() != PFNC_BGR8)
	{
		throw exception("Only BGR8 pixel format is supported.");
	}

	if (cv_image != NULL)
	{
		delete cv_image;
	}
	cv_image = new GpuMat(((int)image->GetHeight()), ((int)image->GetWidth()), CV_8UC3); //, (void*)image->GetData());
	Mat mat_bayer(((int)image->GetHeight()), ((int)image->GetWidth()), CV_8UC3, (void*)image->GetData());
	cv_image->upload(mat_bayer);
	m_device->RequeueBuffer(image);
	return cv_image;
}

cv::cuda::GpuMat CameraControl::GetMat()
{
	return *GetCompleteMat();
}

void CameraControl::SetupCamera(int64_t width, int64_t height)
{
	if (!m_device->IsConnected())
	{
		throw exception("Lost connection to device.");
	}

	SetNodeValue<gcstring>(m_node_map, "ExposureAuto", "Continuous");
	SetNodeValue<int64_t>(m_node_map, "Width", width);
	SetNodeValue<int64_t>(m_node_map, "Height", height);
	SetNodeValue<gcstring>(m_node_map, "AcquisitionMode", "Continuous");
	SetNodeValue<gcstring>(m_node_map, "PixelFormat", "BGR8");

	INodeMap* stream_node_map;
	stream_node_map = m_device->GetTLStreamNodeMap();
	SetNodeValue<gcstring>(stream_node_map, "StreamBufferHandlingMode", "NewestOnly");
}

void CameraControl::EnumerateCamera()
{
	try
	{
		// Get the system
		cout << "Opening system." << endl;
		// Get the devices
		m_system = OpenSystem();

		cout << "Updating devices." << endl;
		m_system->UpdateDevices(1000);

		cout << "Getting devices." << endl;
		std::vector<Arena::DeviceInfo> devices = m_system->GetDevices();

		if (devices.size() != 1)
		{
			throw exception(
				string("Exactly 1 device must be connected. Currently there are " + to_string(devices.size()) + ".")
					.c_str());
		}
		cout << "Creating device." << endl;
		m_device = m_system->CreateDevice(devices[0]);

		cout << "Getting node map." << endl;
		m_node_map = m_device->GetNodeMap();
	}
	catch (GenICam::GenericException& ge)
	{
		std::cout << "\nGenICam exception thrown: " << ge.what() << "\n";
		return;
	}
	catch (std::exception& ex)
	{
		std::cout << "Standard exception thrown: " << ex.what() << "\n";
		return;
	}
	catch (...)
	{
		std::cout << "Unexpected exception thrown\n";
		return;
	}
}

void CameraControl::StartAcquisition()
{
	m_device->StartStream();
}

void CameraControl::StopAcquisition()
{
	m_device->StopStream();
}
