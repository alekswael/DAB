# Label Studio Setup Guide

### Prerequisites

1. Refer to the [🔧 Setup](../README.md#-setup) section in the main README file for initial environment preparation.

    NOTE: This guide refers to version 1.16.0 of Label Studio. If you encounter issues, you can install the newest version of Label Studio by following the [official installation guide](https://labelstud.io/guide/install.html).

### Step 1: Start Label Studio

1. Run the following command:

    ```bash
    label-studio start
    ```

    This will launch the Label Studio web interface, accessible through your browser at `http://localhost:8080`.

### Step 2: Login
1. Sign-up to login.

### Step 3: Create a New Project
1. Click on **Create Project**.
2. Enter a project name and description, then click **Save**.

### Step 4: Import Data
1. Go to the **Data Import** tab.
2. Click **Upload files** and upload your pre-annotated dataset.

### Step 5: Configure Labeling Interface
1. Go to the **Labeling Setup** tab.
2. In the left-hand side of the UI, click on **Custom template**.
3. Under **Labeling Interface**, select the **Code** view, and paste the contents of the `labeling_config.xml` file from the `annotation/` directly into the editor.
4. Click **Save** to apply the configuration.

### Step 6: Start Labeling
1. Make sure to thoroughly read the [DAB Annotation Guidelines](annotation/DAB_Annotation_Guidelines.pdf).
2. Navigate to your project.
3. Click on **Label All Tasks**.
4. Begin labeling your data using the configured interface.
5. Save your progress as you work.

### Step 7: Export annotations
1. Press **Export** from the project view.
2. Export as JSON (top-most option).
3. Save in `data/` folder.

### Additional Resources
- [Label Studio Documentation](https://labelstud.io/documentation.html)
- [Label Studio GitHub Repository](https://github.com/heartexlabs/label-studio)