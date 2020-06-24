using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Linq;
using UnityEngine;

public class SocketClient : MonoBehaviour
{

    TcpClient clientSocket = new TcpClient();
    [SerializeField] string host = "127.0.0.1";
    [SerializeField] int port = 1234;

    const string BEGINSTREAM = "begin_stream";
    const string ENDSTREAM = "end_stream";

    const string POSEID = "pose_id";

    List<PoseRig> pose_rigs = new List<PoseRig>();

    [SerializeField] PoseRig poseTemplate;

    void Start()
    {
        clientSocket.Connect(host, port);
        Debug.Log("connected");
    }

    void Update()
    {
        ParseData(ReadDataFromTcp());

        // Debug.Log(poses.Count);
    }

    string ReadDataFromTcp()
    {
        NetworkStream serverStream = clientSocket.GetStream();
        byte[] readStream = new byte[10025];
        serverStream.Read(readStream, 0, 10025);
        string readData = Encoding.ASCII.GetString(readStream);
        return readData;
    }

    

    void ParseData(string data)
    {
        List<posedata> new_poses = new List<posedata>();

        string[] splitstr = data.Split(';');
        if (splitstr[0] != BEGINSTREAM || splitstr.Length == 0)
            return;

        // posedata current_pose = new posedata();
        PoseRig current_pose = new PoseRig();

        foreach (string s in splitstr)
        {
            // posedata current_pose;
            // current_pose.jointPositions = new Vector3[18];

            if (s == BEGINSTREAM || s == ENDSTREAM)
                continue;

            string[] indexsplit = s.Split(':');

            if (indexsplit[0] == POSEID)
            {
                int p_id = int.Parse(indexsplit[1]);
                
                // find if the pose already exists, if not, then create new
                bool found = false;
                foreach (PoseRig p in pose_rigs)
                {
                    if (p.pdata.pose_id == p_id)
                    {
                        current_pose = p;
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    Debug.Log("new pose");
                    current_pose = Instantiate(poseTemplate, Vector3.zero, Quaternion.identity);
                    current_pose.pdata.pose_id = p_id;
                    pose_rigs.Add(current_pose);
                }

                // current_pose.pose_id = int.Parse(indexsplit[1]);
            }

            if (indexsplit[0].Length > 2)
                continue;


            int jointid = 0;
            float x = 0f;
            float y = 0f;
            float z = 0f;

            try
            {
                jointid = int.Parse(indexsplit[0]);
                x = float.Parse(indexsplit[1], System.Globalization.CultureInfo.InvariantCulture);
                y = float.Parse(indexsplit[2], System.Globalization.CultureInfo.InvariantCulture);
                z = float.Parse(indexsplit[3], System.Globalization.CultureInfo.InvariantCulture);


                // if (jointid == 4)
                    // Debug.Log(x.ToString() + " " + y.ToString() + " " + z.ToString());
            }
            catch
            {
                Debug.Log("numeric parse failed on: " + s);
            }

            if (current_pose.pdata.jointPositions == null)
                current_pose.pdata.jointPositions = new Vector3[18];

            current_pose.pdata.jointPositions[jointid] = new Vector3(x, y, z);
        }

        for (int i = 0; i < pose_rigs.Count; i++)
        {
            if (pose_rigs[i].pdata.pose_id == current_pose.pdata.pose_id)
            {
                pose_rigs[i] = current_pose;
                break;
            }
        }
    }

}
