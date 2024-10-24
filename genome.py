import numpy as np
import copy 
import random

class Genome():
    # Custom Code
    @staticmethod
    def get_random_gene(gene_length):
        # Generate a random gene with the given length, setting values according to the gene specification
        spec = Genome.get_gene_spec()
        gene = np.zeros(gene_length)
        # Check if the spec's evolvable flag (true or false)
        for key in spec.keys():
            if spec[key]["evolvable"]:
                ind = spec[key]["ind"]
                gene[ind] = random.random()
        return gene

    
    @staticmethod 
    def get_random_genome(gene_length, gene_count):
        # Generate a random genome consisting of multiple genes
        genome = [Genome.get_random_gene(gene_length) for i in range(gene_count)]
        return genome

    # Custom Code
    @staticmethod
    def get_gene_spec():
        # Define the specification for the genes in the genome
        # Added evolvable flag and several new specs
        gene_spec =  {
            "link-shape": {"scale": 1, "evolvable": False}, 
            "link-length": {"scale": 2, "evolvable": True},
            "link-radius": {"scale": 1, "evolvable": True},
            "link-recurrence": {"scale": 3, "evolvable": False},
            "link-mass": {"scale": 1, "evolvable": True},
            "link-density": {"scale": 5, "evolvable": True},
            "joint-type": {"scale": 1, "evolvable": True},
            "joint-parent": {"scale": 1, "evolvable": False},
            "joint-axis-xyz": {"scale": 1, "evolvable": True},
            "joint-origin-rpy-1": {"scale": np.pi * 2, "evolvable": False},
            "joint-origin-rpy-2": {"scale": np.pi * 2, "evolvable": True},
            "joint-origin-rpy-3": {"scale": np.pi * 2, "evolvable": False},
            "joint-origin-xyz-1": {"scale": 1, "evolvable": True},
            "joint-origin-xyz-2": {"scale": 1, "evolvable": False},
            "joint-origin-xyz-3": {"scale": 1, "evolvable": True},
            "control-waveform": {"scale": 1, "evolvable": True},
            "control-amp": {"scale": 0.25, "evolvable": True},
            "control-freq": {"scale": 1, "evolvable": True},
            "shape-type": {"scale": 3, "evolvable": False}, 
            "shape-dim1": {"scale": 1, "evolvable": True},
            "shape-dim2": {"scale": 1, "evolvable": True},
            "shape-dim3": {"scale": 1, "evolvable": True}
        }

        ind = 0
        for key in gene_spec.keys():
            gene_spec[key]["ind"] = ind
            ind = ind + 1
        return gene_spec
    
    @staticmethod
    def get_gene_dict(gene, spec):
        # Convert a gene into a dictionary based on specification
        gdict = {}
        for key in spec:
            ind = spec[key]["ind"]
            scale = spec[key]["scale"]
            gdict[key] = gene[ind] * scale
        return gdict

    @staticmethod
    def get_genome_dicts(genome, spec):
        # Convert a genenome into a list of dictionaries
        gdicts = []
        for gene in genome:
            gdicts.append(Genome.get_gene_dict(gene, spec))
        return gdicts

    @staticmethod
    def expandLinks(parent_link, uniq_parent_name, flat_links, exp_links):
        # Recursively expand links to create a structure
        children = [l for l in flat_links if l.parent_name == parent_link.name]
        sibling_ind = 1
        for c in children:
            for r in range(int(c.recur)):
                sibling_ind  = sibling_ind +1
                c_copy = copy.copy(c)
                c_copy.parent_name = uniq_parent_name
                uniq_name = c_copy.name + str(len(exp_links))
                c_copy.name = uniq_name
                c_copy.sibling_ind = sibling_ind
                exp_links.append(c_copy)
                assert c.parent_name != c.name, "Genome::expandLinks: link joined to itself: " + c.name + " joins " + c.parent_name 
                Genome.expandLinks(c, uniq_name, flat_links, exp_links)

    # Custom Code
    @staticmethod
    def genome_to_links(gdicts):
        # Convert genome dictionaries to URDFLink objects
        links = []
        link_ind = 0
        parent_names = [str(link_ind)]
        for gdict in gdicts:
            link_name = str(link_ind)
            parent_ind = gdict["joint-parent"] * len(parent_names)
            assert parent_ind < len(parent_names), "genome.py: parent ind too high: " + str(parent_ind) + "got: " + str(parent_names)
            parent_name = parent_names[int(parent_ind)]
            recur = gdict["link-recurrence"]
            link = URDFLink(name=link_name, 
                            parent_name=parent_name, 
                            recur=recur+1, 
                            link_length=gdict["link-length"], 
                            link_radius=gdict["link-radius"], 
                            link_mass=gdict["link-mass"],
                            joint_type=gdict["joint-type"],
                            joint_parent=gdict["joint-parent"],
                            joint_axis_xyz=gdict["joint-axis-xyz"],
                            joint_origin_rpy_1=gdict["joint-origin-rpy-1"],
                            joint_origin_rpy_2=gdict["joint-origin-rpy-2"],
                            joint_origin_rpy_3=gdict["joint-origin-rpy-3"],
                            joint_origin_xyz_1=gdict["joint-origin-xyz-1"],
                            joint_origin_xyz_2=gdict["joint-origin-xyz-2"],
                            joint_origin_xyz_3=gdict["joint-origin-xyz-3"],
                            control_waveform=gdict["control-waveform"],
                            control_amp=gdict["control-amp"],
                            control_freq=gdict["control-freq"],
                            shape_type=gdict["shape-type"],  # Added new attributes
                            shape_dim1=gdict["shape-dim1"],  # Added new attributes
                            shape_dim2=gdict["shape-dim2"],  # Added new attributes
                            shape_dim3=gdict["shape-dim3"])  # Added new attributes
            links.append(link)
            if link_ind != 0:  # don't re-add the first link
                parent_names.append(link_name)
            link_ind = link_ind + 1

        # now just fix the first link so it links to nothing
        links[0].parent_name = "None"
        return links

    @staticmethod
    def crossover(g1, g2):
        # Perform crossover between two genomes to create a new genome
        x1 = random.randint(0, len(g1) - 1)
        x2 = random.randint(0, len(g2) - 1)
        g3 = np.concatenate((g1[x1:], g2[x2:]))
        if len(g3) > len(g1):
            g3 = g3[0:len(g1)]
        return g3

    # Custom Code
    @staticmethod
    def point_mutate(genome, rate, amount):
        # Perform point mutation on a genome
        new_genome = copy.copy(genome)
        spec = Genome.get_gene_spec()
        for gene in new_genome:
            for key in spec.keys():
                if spec[key]["evolvable"]:
                    ind = spec[key]["ind"]
                    if random.random() < rate:
                        gene[ind] += amount * (2 * random.random() - 1)
                    if gene[ind] >= 1.0:
                        gene[ind] = 0.9999
                    if gene[ind] < 0.0:
                        gene[ind] = 0.0
        return new_genome

    @staticmethod
    def shrink_mutate(genome, rate):
        # Perform shrink mutation on a genome
        if len(genome) == 1:
            return copy.copy(genome)
        if random.random() < rate:
            ind = random.randint(0, len(genome) - 1)
            new_genome = np.delete(genome, ind, 0)
            return new_genome
        else:
            return copy.copy(genome)

    @staticmethod
    def grow_mutate(genome, rate):
        # Perform grow mutation on a genome
        if random.random() < rate:
            gene = Genome.get_random_gene(len(genome[0]))
            new_genome = copy.copy(genome)
            new_genome = np.append(new_genome, [gene], axis=0)
            return new_genome
        else:
            return copy.copy(genome)

    # Custom Code
    @staticmethod
    def to_csv(dna, csv_file):
        # Save the genome to csv
        csv_str = ""
        for gene in dna:
            csv_str += ",".join(map(str, gene)) + "\n"

        with open(csv_file, 'w') as f:
            f.write(csv_str)

    @staticmethod
    def from_csv(filename):
        # Load the genome from csv
        csv_str = ''
        with open(filename) as f:
            csv_str = f.read()
        dna = []
        lines = csv_str.split('\n')
        for line in lines:
            vals = line.split(',')
            gene = [float(v) for v in vals if v != '']
            if len(gene) > 0:
                dna.append(gene)
        return dna

class URDFLink:
    # Custom Code
    def __init__(self, name, parent_name, recur, 
                link_length=0.1, 
                link_radius=0.1, 
                link_mass=0.1,
                joint_type=0.1,
                joint_parent=0.1,
                joint_axis_xyz=0.1,
                joint_origin_rpy_1=0.1,
                joint_origin_rpy_2=0.1,
                joint_origin_rpy_3=0.1,
                joint_origin_xyz_1=0.1,
                joint_origin_xyz_2=0.1,
                joint_origin_xyz_3=0.1,
                control_waveform=0.1,
                control_amp=0.1,
                control_freq=0.1,
                shape_type=0.1,  # New attribute for shape type
                shape_dim1=0.1,  # New attribute for shape dimension 1
                shape_dim2=0.1,  # New attribute for shape dimension 2
                shape_dim3=0.1): # New attribute for shape dimension 3
        self.name = name
        self.parent_name = parent_name
        self.recur = recur 
        self.link_length=link_length 
        self.link_radius=link_radius
        self.link_mass=link_mass
        self.joint_type=joint_type
        self.joint_parent=joint_parent
        self.joint_axis_xyz=joint_axis_xyz
        self.joint_origin_rpy_1=joint_origin_rpy_1
        self.joint_origin_rpy_2=joint_origin_rpy_2
        self.joint_origin_rpy_3=joint_origin_rpy_3
        self.joint_origin_xyz_1=joint_origin_xyz_1
        self.joint_origin_xyz_2=joint_origin_xyz_2
        self.joint_origin_xyz_3=joint_origin_xyz_3
        self.control_waveform=control_waveform
        self.control_amp=control_amp
        self.control_freq=control_freq
        self.shape_type = shape_type  # New attribute
        self.shape_dim1 = shape_dim1  # New attribute
        self.shape_dim2 = shape_dim2  # New attribute
        self.shape_dim3 = shape_dim3  # New attribute
        self.sibling_ind = 1

    # Custom Code
    def to_link_element(self, adom):
        # Create an XML element for the link
        link_tag = adom.createElement("link")
        link_tag.setAttribute("name", self.name)
        vis_tag = adom.createElement("visual")
        geom_tag = adom.createElement("geometry")
        
        # Determine the shape of the link
        if self.shape_type < 1.0:
            cyl_tag = adom.createElement("cylinder")
            cyl_tag.setAttribute("length", str(self.link_length))
            cyl_tag.setAttribute("radius", str(self.link_radius))
            geom_tag.appendChild(cyl_tag)
        elif self.shape_type < 2.0:
            box_tag = adom.createElement("box")
            box_tag.setAttribute("size", f"{self.shape_dim1} {self.shape_dim2} {self.shape_dim3}")
            geom_tag.appendChild(box_tag)
        else:
            sph_tag = adom.createElement("sphere")
            sph_tag.setAttribute("radius", str(self.shape_dim1))
            geom_tag.appendChild(sph_tag)
        
        vis_tag.appendChild(geom_tag)
        
        coll_tag = adom.createElement("collision")
        c_geom_tag = adom.createElement("geometry")
        
        # Determine the collision shape
        if self.shape_type < 1.0:
            c_cyl_tag = adom.createElement("cylinder")
            c_cyl_tag.setAttribute("length", str(self.link_length))
            c_cyl_tag.setAttribute("radius", str(self.link_radius))
            c_geom_tag.appendChild(c_cyl_tag)
        elif self.shape_type < 2.0:
            c_box_tag = adom.createElement("box")
            c_box_tag.setAttribute("size", f"{self.shape_dim1} {self.shape_dim2} {self.shape_dim3}")
            c_geom_tag.appendChild(c_box_tag)
        else:
            c_sph_tag = adom.createElement("sphere")
            c_sph_tag.setAttribute("radius", str(self.shape_dim1))
            c_geom_tag.appendChild(c_sph_tag)
        
        coll_tag.appendChild(c_geom_tag)
        
        # Set the inertial properties of the link
        inertial_tag = adom.createElement("inertial")
        mass_tag = adom.createElement("mass")
        mass = np.pi * (self.link_radius * self.link_radius) * self.link_length
        mass_tag.setAttribute("value", str(mass))
        inertia_tag = adom.createElement("inertia")
        inertia_tag.setAttribute("ixx", "0.03")
        inertia_tag.setAttribute("iyy", "0.03")
        inertia_tag.setAttribute("izz", "0.03")
        inertia_tag.setAttribute("ixy", "0")
        inertia_tag.setAttribute("ixz", "0")
        inertia_tag.setAttribute("iyx", "0")
        inertial_tag.appendChild(mass_tag)
        inertial_tag.appendChild(inertia_tag)
        
        link_tag.appendChild(vis_tag)
        link_tag.appendChild(coll_tag)
        link_tag.appendChild(inertial_tag)
        
        return link_tag

    def to_joint_element(self, adom):
        # Create an XML element for the joint
        joint_tag = adom.createElement("joint")
        joint_tag.setAttribute("name", self.name + "_to_" + self.parent_name)
        if self.joint_type >= 0.5:
            joint_tag.setAttribute("type", "revolute")
        else:
            joint_tag.setAttribute("type", "revolute")
        parent_tag = adom.createElement("parent")
        parent_tag.setAttribute("link", self.parent_name)
        child_tag = adom.createElement("child")
        child_tag.setAttribute("link", self.name)
        axis_tag = adom.createElement("axis")
        if self.joint_axis_xyz <= 0.33:
            axis_tag.setAttribute("xyz", "1 0 0")
        if self.joint_axis_xyz > 0.33 and self.joint_axis_xyz <= 0.66:
            axis_tag.setAttribute("xyz", "0 1 0")
        if self.joint_axis_xyz > 0.66:
            axis_tag.setAttribute("xyz", "0 0 1")
        
        limit_tag = adom.createElement("limit")
        limit_tag.setAttribute("effort", "1")
        limit_tag.setAttribute("upper", "-3.1415")
        limit_tag.setAttribute("lower", "3.1415")
        limit_tag.setAttribute("velocity", "1")
        
        orig_tag = adom.createElement("origin")
        
        # Set the origin of the joint
        rpy1 = self.joint_origin_rpy_1 * self.sibling_ind
        rpy = str(rpy1) + " " + str(self.joint_origin_rpy_2) + " " + str(self.joint_origin_rpy_3)
        
        orig_tag.setAttribute("rpy", rpy)
        xyz = str(self.joint_origin_xyz_1) + " " + str(self.joint_origin_xyz_2) + " " + str(self.joint_origin_xyz_3)
        orig_tag.setAttribute("xyz", xyz)

        joint_tag.appendChild(parent_tag)
        joint_tag.appendChild(child_tag)
        joint_tag.appendChild(axis_tag)
        joint_tag.appendChild(limit_tag)
        joint_tag.appendChild(orig_tag)
        return joint_tag