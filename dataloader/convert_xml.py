import os
from lxml import etree
import re
from copy import deepcopy
import mitsuba as mi
from colorama import Fore

mi.set_variant("cuda_ad_rgb")


def createNewXML(original_xml_path, new_xml_path):
    """
    Create a new XML file consumable by Mitsuba 3 describing the same scene with the original XML file.
    """
    xml_dir = os.path.dirname(original_xml_path)
    # Parse the XML files
    tree = etree.parse(original_xml_path)
    root = tree.getroot()
    root.attrib.clear()
    root.set("version", "3.0.0")

    # Modify the XML elements
    for bsdf in root.findall('.//bsdf[@type="microfacet"]'):
        bsdf_id = bsdf.get("id")
        uvscale_value = bsdf.find('float[@name="uvScale"]').get("value")
        albedo_filename = bsdf.find('texture[@name="albedo"]/string[@name="filename"]').get("value")
        normal_filename = bsdf.find('texture[@name="normal"]/string[@name="filename"]').get("value")
        roughness_filename = bsdf.find('texture[@name="roughness"]/string[@name="filename"]').get(
            "value"
        )

        # Replace the path before "Material..." with "data/SubstanceBRDF"
        albedo_filename = re.sub(r"^(.+?)?(?=Material)", "data/SubstanceBRDF/", albedo_filename)
        normal_filename = re.sub(r"^(.+/)?(?=Material)", "data/SubstanceBRDF/", normal_filename)
        roughness_filename = re.sub(
            r"^(.+/)?(?=Material)", "data/SubstanceBRDF/", roughness_filename
        )

        # Create the new XML structure
        new_bsdf = etree.Element("bsdf", attrib={"id": bsdf_id, "type": "twosided"})

        def uvTransformElement(uvscale_value):
            uv_transform = etree.Element("transform", attrib={"name": "to_uv"})
            scale = etree.Element("scale", attrib={"x": uvscale_value, "y": uvscale_value})
            uv_transform.append(scale)
            return uv_transform

        normalmap_bsdf = etree.Element("bsdf", attrib={"type": "normalmap"})
        normalmap_texture = etree.Element("texture", attrib={"name": "normalmap", "type": "bitmap"})
        normalmap_filename = etree.Element(
            "string", attrib={"name": "filename", "value": normal_filename}
        )
        normalmap_texture.append(etree.Element("boolean", attrib={"name": "raw", "value": "true"}))
        normalmap_texture.append(normalmap_filename)
        normalmap_texture.append(uvTransformElement(uvscale_value))
        normalmap_bsdf.append(normalmap_texture)

        principled_bsdf = etree.Element("bsdf", attrib={"type": "principled"})

        base_color_texture = etree.Element(
            "texture", attrib={"name": "base_color", "type": "bitmap"}
        )
        base_color_filename = etree.Element(
            "string", attrib={"name": "filename", "value": albedo_filename}
        )
        base_color_texture.append(base_color_filename)
        base_color_texture.append(uvTransformElement(uvscale_value))
        principled_bsdf.append(base_color_texture)

        roughness_texture = etree.Element("texture", attrib={"name": "roughness", "type": "bitmap"})
        roughness_filename_elem = etree.Element(
            "string", attrib={"name": "filename", "value": roughness_filename}
        )
        roughness_texture.append(roughness_filename_elem)
        roughness_texture.append(uvTransformElement(uvscale_value))
        principled_bsdf.append(roughness_texture)

        normalmap_bsdf.append(principled_bsdf)
        new_bsdf.append(normalmap_bsdf)

        # Replace the old bsdf with the new one
        bsdf.getparent().replace(bsdf, new_bsdf)

    # shape
    shape_id_cnts = {}
    for shape in root.findall(".//shape"):
        # Rename the shape ids to avoid duplicates
        shape_id = shape.get("id")
        if shape_id in shape_id_cnts:
            shape.set("id", f"{shape_id}_{shape_id_cnts[shape_id]}")
            shape_id_cnts[shape_id] += 1
            shape_id = shape.get("id")
        else:
            shape_id_cnts[shape_id] = 1

        toWorld = shape.find('transform[@name="toWorld"]')
        if toWorld is not None:
            toWorld.set("name", "to_world")

        if shape.get("type") == "obj":
            obj_filename = shape.find('string[@name="filename"]').get("value")
            obj_filename = re.sub(r"^(\.\.\/)+", "data/", obj_filename)

            if len(shape.findall('ref[@name="bsdf"]')) > 1:
                # open obj file and split it according to the material
                # keep the content before the first usemtl as shared part in all obj files
                # then split the obj file according to the g or usemtl, whichever comes first
                # and save it in the same directory as the xml file
                # and replace the filename with the new path
                with open(obj_filename, "r") as f:
                    obj_content = f.read()
                    if re.search(r"\ng", obj_content) is not None:
                        found_g = True
                        sections = re.split(r"\ng ", obj_content)
                    else:
                        found_g = False
                        sections = re.split(r"\nusemtl ", obj_content)
                    shared_content = sections[0]
                    for i, section in enumerate(sections[1:]):
                        if found_g:
                            section = "g " + section
                        else:
                            section = "usemtl " + section
                        mtl_name = re.search(r"usemtl (.+)", section).group(1)
                        new_obj_filename = os.path.join(
                            re.sub(r"^data", xml_dir, os.path.dirname(obj_filename)),
                            os.path.splitext(os.path.basename(obj_filename))[0]
                            + "_"
                            + mtl_name
                            + ".obj",
                        )
                        os.makedirs(os.path.dirname(new_obj_filename), exist_ok=True)
                        with open(
                            new_obj_filename,
                            "w",
                        ) as f:
                            f.write(shared_content + "\n" + section)

                        new_shape = etree.Element(
                            "shape", attrib={"type": "obj", "id": f"{shape_id}_{mtl_name}"}
                        )
                        new_shape.append(
                            etree.Element(
                                "string", attrib={"name": "filename", "value": new_obj_filename}
                            )
                        )
                        new_shape.append(
                            etree.Element("ref", attrib={"name": "bsdf", "id": mtl_name})
                        )
                        new_shape.append(deepcopy(toWorld))
                        root.append(new_shape)
                shape.getparent().remove(shape)

            shape.find('string[@name="filename"]').set("value", obj_filename)

    # integrator
    integrator = root.find(".//integrator")
    if integrator is not None:
        new_integrator = etree.Element("integrator", attrib={"type": "prb"})
        new_integrator.append(etree.Element("integer", attrib={"name": "max_depth", "value": "6"}))
        integrator.getparent().replace(integrator, new_integrator)

    # sensor
    sensor = root.find(".//sensor")
    if sensor is not None:
        sensor.set("id", "sensor")
        fovAxis = sensor.find(".//string[@name='fovAxis']")
        if fovAxis is not None:
            fovAxis.set("name", "fov_axis")
        sampler = sensor.find(".//sampler")
        sampler.set("type", "multijitter")
        sampleCount = sampler.find(".//integer[@name='sampleCount']")
        if sampleCount is not None:
            sampleCount.set("name", "sample_count")

    # env emitter
    emitter = root.find(".//emitter[@type='envmap']")
    if emitter is not None:
        env_filename_element = emitter.find(".//string[@name='filename']")
        env_filename = env_filename_element.get("value")
        env_filename = re.sub(r"^(.+?)?(?=EnvDataset)", "data/", env_filename)
        env_filename_element.set("value", env_filename)

    # area emitter
    area_emitters = root.findall(".//emitter[@type='area']")
    for area_emitter in area_emitters:
        area_emitter.find(".//rgb").set("name", "radiance")

    # Write the modified XML back to the file
    tree.write(new_xml_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")


def convertXML2Dict(xml_path, tmp_dir="tmp/"):
    """
    Convert an XML file to a Python dict consumable by Mistuba 3.
    """
    xml_dir = os.path.dirname(xml_path)
    # Parse the XML files
    tree = etree.parse(xml_path)
    root = tree.getroot()

    scene_dict = {"type": "scene"}

    # Modify the XML elements
    for bsdf in root.findall('.//bsdf[@type="microfacet"]'):
        bsdf_id = bsdf.get("id")
        uvscale_value = float(bsdf.find('float[@name="uvScale"]').get("value"))
        albedo_filename = bsdf.find('texture[@name="albedo"]/string[@name="filename"]').get("value")
        normal_filename = bsdf.find('texture[@name="normal"]/string[@name="filename"]').get("value")
        roughness_filename = bsdf.find('texture[@name="roughness"]/string[@name="filename"]').get(
            "value"
        )

        # Replace the path before "Material..." with "data/SubstanceBRDF"
        albedo_filename = re.sub(r"^(.+?)?(?=Material)", "data/SubstanceBRDF/", albedo_filename)
        normal_filename = re.sub(r"^(.+/)?(?=Material)", "data/SubstanceBRDF/", normal_filename)
        roughness_filename = re.sub(
            r"^(.+/)?(?=Material)", "data/SubstanceBRDF/", roughness_filename
        )

        # Create the new XML structure
        new_bsdf = etree.Element("bsdf", attrib={"id": bsdf_id, "type": "twosided"})
        new_bsdf = {
            "type": "twosided",
            "material": {
                "type": "normalmap",
                "normalmap": {"type": "bitmap", "raw": True, "filename": normal_filename},
                "bsdf": {
                    "type": "principled",
                    "base_color": {
                        "type": "bitmap",
                        "filename": albedo_filename,
                        "to_uv": mi.ScalarTransform4f.scale([uvscale_value, uvscale_value, 1]),
                    },
                    "roughness": {
                        "type": "bitmap",
                        "filename": roughness_filename,
                        "to_uv": mi.ScalarTransform4f.scale([uvscale_value, uvscale_value, 1]),
                    },
                },
            },
        }

        scene_dict[bsdf_id] = new_bsdf

    def parseTransform(transform):
        transform_dict = {}
        for child in transform:
            transform_dict[child.tag] = [float(x) for x in child.values()]
        return transform_dict

    # shape
    shape_id_cnts = {}
    for shape in root.findall(".//shape"):
        # Rename the shape ids to avoid duplicates
        shape_id = shape.get("id")
        if shape_id in shape_id_cnts:
            shape.set("id", f"{shape_id}_{shape_id_cnts[shape_id]}")
            shape_id_cnts[shape_id] += 1
            shape_id = shape.get("id")
        else:
            shape_id_cnts[shape_id] = 1

        to_world = mi.ScalarTransform4f()
        toWorld = shape.find('transform[@name="toWorld"]')
        if toWorld is not None:
            for transform in toWorld.xpath("./*"):
                if transform.tag == "rotate":
                    x = float(transform.get("x"))
                    y = float(transform.get("y"))
                    z = float(transform.get("z"))
                    angle = float(transform.get("angle"))
                    matrix = eval(
                        f"mi.ScalarTransform4f.{transform.tag}(axis={[x, y, z]}, angle={angle})"
                    )
                else:
                    x = float(transform.get("x"))
                    y = float(transform.get("y"))
                    z = float(transform.get("z"))
                    matrix = eval(f"mi.ScalarTransform4f.{transform.tag}({[x, y, z]})")
                to_world = matrix @ to_world

        if shape.get("type") == "obj":
            obj_filename = shape.find('string[@name="filename"]').get("value")
            obj_filename = re.sub(r"^(\.\.\/)+", "data/", obj_filename)

            if len(shape.findall('ref[@name="bsdf"]')) > 1:
                # open obj file and split it according to the material
                # keep the content before the first usemtl as shared part in all obj files
                # then split the obj file according to the g or usemtl, whichever comes first
                # and save it in the same directory as the xml file
                # and replace the filename with the new path
                with open(obj_filename, "r") as f:
                    obj_content = f.read()
                    if re.search(r"\ng", obj_content) is not None:
                        found_g = True
                        sections = re.split(r"\ng ", obj_content)
                    else:
                        found_g = False
                        sections = re.split(r"\nusemtl ", obj_content)
                    shared_content = sections[0]
                    for i, section in enumerate(sections[1:]):
                        if found_g:
                            section = "g " + section
                        else:
                            section = "usemtl " + section
                        mtl_name = re.search(r"usemtl (.+)", section).group(1)
                        new_obj_filename = os.path.join(
                            re.sub(r"^data", tmp_dir, os.path.dirname(obj_filename)),
                            os.path.splitext(os.path.basename(obj_filename))[0]
                            + "_"
                            + mtl_name
                            + ".obj",
                        )
                        os.makedirs(os.path.dirname(new_obj_filename), exist_ok=True)
                        with open(
                            new_obj_filename,
                            "w",
                        ) as f:
                            f.write(shared_content + "\n" + section)

                        new_shape_id = f"{shape_id}_{mtl_name}"
                        new_shape = {
                            "type": "obj",
                            "filename": new_obj_filename,
                            "to_world": to_world,
                            "bsdf": {"type": "ref", "id": mtl_name},
                        }

                        scene_dict[new_shape_id] = new_shape
            else:
                new_shape = {
                    "type": "obj",
                    "filename": obj_filename,
                    "to_world": to_world,
                }

                bsdf = shape.find('ref[@name="bsdf"]')
                if bsdf is not None:
                    bsdf_id = bsdf.get("id")
                    new_shape["bsdf"] = {"type": "ref", "id": bsdf_id}

                emitter = shape.find("emitter")
                if emitter is not None:
                    radiance = emitter.find("rgb").get("value")
                    radiance = [float(x) for x in radiance.split()]

                    new_shape["emitter"] = {
                        "type": emitter.get("type"),
                        "radiance": {"type": "rgb", "value": radiance},
                    }

                scene_dict[shape_id] = new_shape
        else:
            print(Fore.YELLOW + "Warning: Found non-obj shape.")

    # integrator
    integrator = root.find(".//integrator")
    if integrator is not None:
        new_integrator = {"type": "path", "max_depth": 6}
        scene_dict["integrator"] = new_integrator

    # sensor
    sensor = root.find(".//sensor")
    if sensor is not None:
        fov = float(sensor.find(".//float[@name='fov']").get("value"))
        fov_axis = sensor.find(".//string[@name='fovAxis']").get("value")
        film = sensor.find(".//film")
        sampler = sensor.find(".//sampler")
        sample_count = int(sampler.find(".//integer[@name='sampleCount']").get("value"))
        new_sensor = {
            "type": "perspective",
            "fov": fov,
            "fov_axis": fov_axis,
            "film": {
                "type": film.get("type"),
                "width": int(film.find(".//integer[@name='width']").get("value")),
                "height": int(film.find(".//integer[@name='height']").get("value")),
                'sample_border': True,
            },
            "sampler": {"type": "multijitter", "sample_count": sample_count},
        }
        scene_dict["sensor"] = new_sensor

    # env emitter
    emitter = root.find(".//emitter[@type='envmap']")
    if emitter is not None:
        env_filename = emitter.find(".//string[@name='filename']").get("value")
        env_filename = re.sub(r"^(.+?)?(?=EnvDataset)", "data/", env_filename)
        scale = float(emitter.find(".//float[@name='scale']").get("value"))
        new_env_emitter = {
            "type": "envmap",
            "filename": env_filename,
            "scale": scale,
        }
        scene_dict["env_emitter"] = new_env_emitter

    return scene_dict

